package com.anomanet.transaction.neo4j;

import com.anomanet.transaction.model.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.neo4j.core.Neo4jClient;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class GraphWriterService {

    private static final Logger log = LoggerFactory.getLogger(GraphWriterService.class);
    private final Neo4jClient neo4jClient;

    public GraphWriterService(Neo4jClient neo4jClient) {
        this.neo4jClient = neo4jClient;
    }

    public void writeTransactionEdge(Transaction tx) {
        try {
            String cypher = """
                MERGE (src:Account {id: $srcId})
                ON CREATE SET src.branch_id = $branchId, src.anoma_score = 0.0, src.is_dormant = false
                MERGE (dst:Account {id: $dstId})
                ON CREATE SET dst.branch_id = $branchId, dst.anoma_score = 0.0, dst.is_dormant = false
                CREATE (src)-[:TRANSFERRED_TO {
                    amount: $amount,
                    timestamp: $timestamp,
                    channel: $channel,
                    tx_id: $txId,
                    branch_id: $branchId
                }]->(dst)
                """;

            neo4jClient.query(cypher)
                .bindAll(Map.of(
                    "srcId",     tx.getSourceAccountId(),
                    "dstId",     tx.getDestAccountId(),
                    "branchId",  tx.getBranchId() != null ? tx.getBranchId() : "UNKNOWN",
                    "amount",    tx.getAmount().doubleValue(),
                    "timestamp", tx.getInitiatedAt().toString(),
                    "channel",   tx.getChannel().name(),
                    "txId",      tx.getId().toString()
                ))
                .run();

            log.debug("Graph edge written: {} -> {}", tx.getSourceAccountId(), tx.getDestAccountId());
        } catch (Exception e) {
            log.error("Failed to write graph edge for tx: {}", tx.getId(), e);
        }
    }
}
