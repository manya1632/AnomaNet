package com.anomanet.transaction.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "transactions")
public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "reference_number", unique = true, nullable = false)
    private String referenceNumber;

    @Column(name = "source_account_id", nullable = false)
    private String sourceAccountId;

    @Column(name = "dest_account_id", nullable = false)
    private String destAccountId;

    @Column(nullable = false, precision = 18, scale = 2)
    private BigDecimal amount;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Channel channel;

    @Column(name = "initiated_at", nullable = false)
    private Instant initiatedAt;

    @Column(name = "settled_at")
    private Instant settledAt;

    @Column(name = "branch_id")
    private String branchId;

    @Enumerated(EnumType.STRING)
    private Status status = Status.SETTLED;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();

    public enum Channel { NEFT, RTGS, IMPS, UPI, SWIFT, CASH, BRANCH }
    public enum Status  { PENDING, SETTLED, FAILED, REVERSED }

    public UUID getId() { return id; }
    public String getReferenceNumber() { return referenceNumber; }
    public void setReferenceNumber(String v) { referenceNumber = v; }
    public String getSourceAccountId() { return sourceAccountId; }
    public void setSourceAccountId(String v) { sourceAccountId = v; }
    public String getDestAccountId() { return destAccountId; }
    public void setDestAccountId(String v) { destAccountId = v; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal v) { amount = v; }
    public Channel getChannel() { return channel; }
    public void setChannel(Channel v) { channel = v; }
    public Instant getInitiatedAt() { return initiatedAt; }
    public void setInitiatedAt(Instant v) { initiatedAt = v; }
    public Instant getSettledAt() { return settledAt; }
    public void setSettledAt(Instant v) { settledAt = v; }
    public String getBranchId() { return branchId; }
    public void setBranchId(String v) { branchId = v; }
    public Status getStatus() { return status; }
    public void setStatus(Status v) { status = v; }
    public Instant getCreatedAt() { return createdAt; }
}
