package com.anomanet.graph.dto;

import java.util.List;

public class GraphDtos {

    public static class SubgraphRequest {
        private String accountId;
        private int depth = 2;
        private int hours = 168;
        public String getAccountId() { return accountId; }
        public void setAccountId(String v) { accountId = v; }
        public int getDepth() { return depth; }
        public void setDepth(int v) { depth = v; }
        public int getHours() { return hours; }
        public void setHours(int v) { hours = v; }
    }

    public static class CycleRequest {
        private String accountId;
        private int maxLength = 7;
        private int hours = 168;
        public String getAccountId() { return accountId; }
        public void setAccountId(String v) { accountId = v; }
        public int getMaxLength() { return maxLength; }
        public void setMaxLength(int v) { maxLength = v; }
        public int getHours() { return hours; }
        public void setHours(int v) { hours = v; }
    }

    public static class PathRequest {
        private String sourceId;
        private String destId;
        private int maxHops = 5;
        public String getSourceId() { return sourceId; }
        public void setSourceId(String v) { sourceId = v; }
        public String getDestId() { return destId; }
        public void setDestId(String v) { destId = v; }
        public int getMaxHops() { return maxHops; }
        public void setMaxHops(int v) { maxHops = v; }
    }

    // THE critical response shape — agreed with Manya's FundFlowGraph
    public static class SubgraphResponse {
        private List<GraphNode> nodes;
        private List<GraphEdge> edges;
        private GraphMetadata metadata;
        public List<GraphNode> getNodes() { return nodes; }
        public void setNodes(List<GraphNode> v) { nodes = v; }
        public List<GraphEdge> getEdges() { return edges; }
        public void setEdges(List<GraphEdge> v) { edges = v; }
        public GraphMetadata getMetadata() { return metadata; }
        public void setMetadata(GraphMetadata v) { metadata = v; }
    }

    public static class GraphNode {
        private String id;
        private String label;
        private double anomaScore;
        private String accountType;
        private boolean isDormant;
        private String kycRiskTier;
        private String branchId;
        public String getId() { return id; }
        public void setId(String v) { id = v; }
        public String getLabel() { return label; }
        public void setLabel(String v) { label = v; }
        public double getAnomaScore() { return anomaScore; }
        public void setAnomaScore(double v) { anomaScore = v; }
        public String getAccountType() { return accountType; }
        public void setAccountType(String v) { accountType = v; }
        public boolean isDormant() { return isDormant; }
        public void setDormant(boolean v) { isDormant = v; }
        public String getKycRiskTier() { return kycRiskTier; }
        public void setKycRiskTier(String v) { kycRiskTier = v; }
        public String getBranchId() { return branchId; }
        public void setBranchId(String v) { branchId = v; }
    }

    public static class GraphEdge {
        private String source;
        private String target;
        private double amount;
        private String timestamp;
        private String channel;
        private String txId;
        public String getSource() { return source; }
        public void setSource(String v) { source = v; }
        public String getTarget() { return target; }
        public void setTarget(String v) { target = v; }
        public double getAmount() { return amount; }
        public void setAmount(double v) { amount = v; }
        public String getTimestamp() { return timestamp; }
        public void setTimestamp(String v) { timestamp = v; }
        public String getChannel() { return channel; }
        public void setChannel(String v) { channel = v; }
        public String getTxId() { return txId; }
        public void setTxId(String v) { txId = v; }
    }

    public static class GraphMetadata {
        private int totalNodes;
        private int totalEdges;
        private List<List<String>> detectedCycles;
        public int getTotalNodes() { return totalNodes; }
        public void setTotalNodes(int v) { totalNodes = v; }
        public int getTotalEdges() { return totalEdges; }
        public void setTotalEdges(int v) { totalEdges = v; }
        public List<List<String>> getDetectedCycles() { return detectedCycles; }
        public void setDetectedCycles(List<List<String>> v) { detectedCycles = v; }
    }

    public static class CycleResult {
        private List<String> path;
        private double amountVariance;
        private double completionHours;
        public List<String> getPath() { return path; }
        public void setPath(List<String> v) { path = v; }
        public double getAmountVariance() { return amountVariance; }
        public void setAmountVariance(double v) { amountVariance = v; }
        public double getCompletionHours() { return completionHours; }
        public void setCompletionHours(double v) { completionHours = v; }
    }

    public static class AccountStats {
        private double centrality;
        private long degreeIn;
        private long degreeOut;
        private String clusterId;
        public double getCentrality() { return centrality; }
        public void setCentrality(double v) { centrality = v; }
        public long getDegreeIn() { return degreeIn; }
        public void setDegreeIn(long v) { degreeIn = v; }
        public long getDegreeOut() { return degreeOut; }
        public void setDegreeOut(long v) { degreeOut = v; }
        public String getClusterId() { return clusterId; }
        public void setClusterId(String v) { clusterId = v; }
    }
}
