package com.anomanet.transaction.kafka;

import com.anomanet.transaction.model.Transaction;
import com.anomanet.transaction.neo4j.GraphWriterService;
import com.anomanet.transaction.repository.TransactionRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

@Component
public class TransactionConsumer {

    private static final Logger log = LoggerFactory.getLogger(TransactionConsumer.class);

    private final TransactionRepository transactionRepository;
    private final GraphWriterService graphWriterService;
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;

    @Value("${kafka.topics.ml-scoring-queue}")
    private String mlScoringTopic;

    public TransactionConsumer(TransactionRepository transactionRepository,
                               GraphWriterService graphWriterService,
                               KafkaTemplate<String, String> kafkaTemplate,
                               ObjectMapper objectMapper) {
        this.transactionRepository = transactionRepository;
        this.graphWriterService = graphWriterService;
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    @KafkaListener(topics = "${kafka.topics.raw-transactions}", groupId = "transaction-service")
    public void consume(String message) {
        try {
            JsonNode event = objectMapper.readTree(message);
            JsonNode txn = event.get("transaction");

            Transaction tx = new Transaction();
            tx.setReferenceNumber(txn.get("reference_number").asText());
            tx.setSourceAccountId(txn.get("source_account_id").asText());
            tx.setDestAccountId(txn.get("dest_account_id").asText());
            tx.setAmount(new BigDecimal(txn.get("amount").asText()));
            tx.setChannel(Transaction.Channel.valueOf(txn.get("channel").asText()));
            tx.setInitiatedAt(Instant.parse(txn.get("initiated_at").asText()));
            tx.setBranchId(txn.has("branch_id") ? txn.get("branch_id").asText() : null);
            tx.setStatus(Transaction.Status.SETTLED);

            Transaction saved = transactionRepository.save(tx);
            log.info("Transaction saved: {} amount={}", saved.getId(), saved.getAmount());

            // Write to Neo4j
            graphWriterService.writeTransactionEdge(saved);

            // Publish enrichment event to ML scoring queue
            JsonNode accountNode = event.has("account") ? event.get("account") : objectMapper.createObjectNode();

            Map<String, Object> enrichment = new HashMap<>();
            enrichment.put("event_type", "TRANSACTION_ENRICHED");
            enrichment.put("event_id", java.util.UUID.randomUUID().toString());
            enrichment.put("schema_version", "1.0");
            enrichment.put("timestamp", Instant.now().toString());

            Map<String, Object> txMap = new HashMap<>();
            txMap.put("id", saved.getId().toString());
            txMap.put("reference_number", saved.getReferenceNumber());
            txMap.put("source_account_id", saved.getSourceAccountId());
            txMap.put("dest_account_id", saved.getDestAccountId());
            txMap.put("amount", saved.getAmount());
            txMap.put("channel", saved.getChannel().name());
            txMap.put("initiated_at", saved.getInitiatedAt().toString());
            txMap.put("branch_id", saved.getBranchId());
            enrichment.put("transaction", txMap);

            Map<String, Object> accountMap = new HashMap<>();
            accountMap.put("kyc_risk_tier", accountNode.has("kyc_risk_tier") ? accountNode.get("kyc_risk_tier").asText() : "LOW");
            accountMap.put("declared_monthly_income", accountNode.has("declared_monthly_income") ? accountNode.get("declared_monthly_income").asDouble() : 0.0);
            accountMap.put("is_dormant", accountNode.has("is_dormant") && accountNode.get("is_dormant").asBoolean());
            enrichment.put("account", accountMap);

            kafkaTemplate.send(mlScoringTopic, saved.getId().toString(), objectMapper.writeValueAsString(enrichment));
            log.info("Published to ml.scoring.queue: {}", saved.getId());

        } catch (Exception e) {
            log.error("Failed to process transaction", e);
        }
    }
}
