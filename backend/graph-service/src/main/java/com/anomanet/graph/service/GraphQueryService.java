package com.anomanet.graph.service;

import com.anomanet.graph.dto.GraphDtos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.neo4j.core.Neo4jClient;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class GraphQueryService {

    private static final Logger log = LoggerFactory.getLogger(GraphQueryService.class);
    private final Neo4jClient neo4jClient;

    public GraphQueryService(Neo4jClient neo4jClient) {
        this.neo4jClient = neo4jClient;
    }

    public GraphDtos.SubgraphResponse getSubgraph(String accountId, int depth, int hours) {
        int d = Math.min(Math.max(depth, 1), 4);

        String cypher = String.format("""
            MATCH (start:Account {id: $accountId})
            OPTIONAL MATCH path = (start)-[:TRANSFERRED_TO*1..%d]-(neighbor:Account)
            WITH collect(DISTINCT start) + collect(DISTINCT neighbor) AS allNodes
            UNWIND allNodes AS n
            WITH collect(DISTINCT n) AS nodes
            UNWIND nodes AS n
            OPTIONAL MATCH (n)-[r:TRANSFERRED_TO]->(m:Account)
            WHERE m IN nodes
            RETURN
              collect(DISTINCT {
                id: n.id,
                anoma_score: coalesce(n.anoma_score, 0.0),
                account_type: coalesce(n.account_type, 'UNKNOWN'),
                is_dormant: coalesce(n.is_dormant, false),
                kyc_risk_tier: coalesce(n.kyc_risk_tier, 'LOW'),
                branch_id: coalesce(n.branch_id, '')
              }) AS nodes,
              collect(DISTINCT {
                source: startNode(r).id,
                target: endNode(r).id,
                amount: coalesce(r.amount, 0.0),
                timestamp: coalesce(r.timestamp, ''),
                channel: coalesce(r.channel, ''),
                tx_id: coalesce(r.tx_id, '')
              }) AS edges
            """, d);

        List<GraphDtos.GraphNode> nodes = new ArrayList<>();
        List<GraphDtos.GraphEdge> edges = new ArrayList<>();

        try {
            neo4jClient.query(cypher)
                .bind(accountId).to("accountId")
                .fetch()
                .first()
                .ifPresent(row -> {
                    parseNodes(row, nodes);
                    parseEdges(row, edges);
                });
        } catch (Exception e) {
            log.error("Subgraph query failed for {}: {}", accountId, e.getMessage());
            // Return root node at minimum
            GraphDtos.GraphNode root = new GraphDtos.GraphNode();
            root.setId(accountId);
            root.setLabel(accountId);
            root.setAnomaScore(0.0);
            root.setAccountType("UNKNOWN");
            nodes.add(root);
        }

        // Ensure root node always present
        if (nodes.stream().noneMatch(n -> accountId.equals(n.getId()))) {
            GraphDtos.GraphNode root = new GraphDtos.GraphNode();
            root.setId(accountId);
            root.setLabel(accountId);
            nodes.add(0, root);
        }

        GraphDtos.GraphMetadata meta = new GraphDtos.GraphMetadata();
        meta.setTotalNodes(nodes.size());
        meta.setTotalEdges(edges.size());
        meta.setDetectedCycles(new ArrayList<>());

        GraphDtos.SubgraphResponse response = new GraphDtos.SubgraphResponse();
        response.setNodes(nodes);
        response.setEdges(edges);
        response.setMetadata(meta);
        return response;
    }

    @SuppressWarnings("unchecked")
    private void parseNodes(Map<String, Object> row, List<GraphDtos.GraphNode> nodes) {
        Object rawNodes = row.get("nodes");
        if (!(rawNodes instanceof List)) return;
        for (Object obj : (List<?>) rawNodes) {
            if (!(obj instanceof Map)) continue;
            Map<String, Object> n = (Map<String, Object>) obj;
            if (n.get("id") == null) continue;
            GraphDtos.GraphNode node = new GraphDtos.GraphNode();
            node.setId(String.valueOf(n.get("id")));
            node.setLabel(String.valueOf(n.get("id")));
            node.setAnomaScore(n.get("anoma_score") != null ? ((Number) n.get("anoma_score")).doubleValue() : 0.0);
            node.setAccountType(String.valueOf(n.getOrDefault("account_type", "UNKNOWN")));
            node.setDormant(Boolean.TRUE.equals(n.get("is_dormant")));
            node.setKycRiskTier(String.valueOf(n.getOrDefault("kyc_risk_tier", "LOW")));
            node.setBranchId(String.valueOf(n.getOrDefault("branch_id", "")));
            nodes.add(node);
        }
    }

    @SuppressWarnings("unchecked")
    private void parseEdges(Map<String, Object> row, List<GraphDtos.GraphEdge> edges) {
        Object rawEdges = row.get("edges");
        if (!(rawEdges instanceof List)) return;
        for (Object obj : (List<?>) rawEdges) {
            if (!(obj instanceof Map)) continue;
            Map<String, Object> e = (Map<String, Object>) obj;
            if (e.get("source") == null || e.get("target") == null) continue;
            GraphDtos.GraphEdge edge = new GraphDtos.GraphEdge();
            edge.setSource(String.valueOf(e.get("source")));
            edge.setTarget(String.valueOf(e.get("target")));
            edge.setAmount(e.get("amount") != null ? ((Number) e.get("amount")).doubleValue() : 0.0);
            edge.setTimestamp(String.valueOf(e.getOrDefault("timestamp", "")));
            edge.setChannel(String.valueOf(e.getOrDefault("channel", "")));
            edge.setTxId(String.valueOf(e.getOrDefault("tx_id", "")));
            edges.add(edge);
        }
    }

    public List<GraphDtos.CycleResult> detectCycles(String accountId, int maxLength, int hours) {
        int ml = Math.min(Math.max(maxLength, 2), 7);
        String cypher = String.format("""
            MATCH path = (start:Account {id: $accountId})-[:TRANSFERRED_TO*2..%d]->(start)
            WITH [n IN nodes(path) | n.id] AS pathIds,
                 [r IN relationships(path) | r.amount] AS amounts
            RETURN pathIds,
                   reduce(mn=amounts[0], a IN amounts | CASE WHEN a < mn THEN a ELSE mn END) AS minAmt,
                   reduce(mx=amounts[0], a IN amounts | CASE WHEN a > mx THEN a ELSE mx END) AS maxAmt,
                   reduce(s=0.0, a IN amounts | s+a)/size(amounts) AS avgAmt
            LIMIT 10
            """, ml);

        List<GraphDtos.CycleResult> results = new ArrayList<>();
        try {
            neo4jClient.query(cypher)
                .bind(accountId).to("accountId")
                .fetch().all()
                .forEach(row -> {
                    GraphDtos.CycleResult r = new GraphDtos.CycleResult();
                    r.setPath((List<String>) row.get("pathIds"));
                    double avg = row.get("avgAmt") != null ? ((Number)row.get("avgAmt")).doubleValue() : 1;
                    double min = row.get("minAmt") != null ? ((Number)row.get("minAmt")).doubleValue() : 0;
                    double max = row.get("maxAmt") != null ? ((Number)row.get("maxAmt")).doubleValue() : 0;
                    r.setAmountVariance(avg > 0 ? (max - min) / avg : 0);
                    r.setCompletionHours(hours);
                    results.add(r);
                });
        } catch (Exception e) {
            log.error("Cycle detection failed: {}", e.getMessage());
        }
        return results;
    }

    public GraphDtos.AccountStats getAccountStats(String accountId) {
        String cypher = """
            MATCH (a:Account {id: $accountId})
            OPTIONAL MATCH (a)-[out:TRANSFERRED_TO]->()
            OPTIONAL MATCH ()-[in:TRANSFERRED_TO]->(a)
            RETURN count(DISTINCT out) AS degreeOut, count(DISTINCT in) AS degreeIn
            """;
        GraphDtos.AccountStats stats = new GraphDtos.AccountStats();
        try {
            neo4jClient.query(cypher)
                .bind(accountId).to("accountId")
                .fetch().first()
                .ifPresent(row -> {
                    stats.setDegreeOut(((Number)row.get("degreeOut")).longValue());
                    stats.setDegreeIn(((Number)row.get("degreeIn")).longValue());
                    stats.setCentrality(0.0);
                    stats.setClusterId("N/A");
                });
        } catch (Exception e) {
            log.error("Stats query failed: {}", e.getMessage());
        }
        return stats;
    }
}
