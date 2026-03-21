package com.anomanet.cases.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "cases")
public class Case {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "alert_id", nullable = false)
    private UUID alertId;

    @Column(name = "assigned_to")
    private UUID assignedTo;

    @Enumerated(EnumType.STRING)
    private CaseStatus status = CaseStatus.OPEN;

    @Enumerated(EnumType.STRING)
    private CasePriority priority = CasePriority.MEDIUM;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();

    @Column(name = "updated_at")
    private Instant updatedAt = Instant.now();

    @Column(name = "closed_at")
    private Instant closedAt;

    public enum CaseStatus  { OPEN, UNDER_REVIEW, ESCALATED, CLOSED_SAR, CLOSED_FP }
    public enum CasePriority { LOW, MEDIUM, HIGH, CRITICAL }

    public UUID getId() { return id; }
    public UUID getAlertId() { return alertId; }
    public void setAlertId(UUID v) { alertId = v; }
    public UUID getAssignedTo() { return assignedTo; }
    public void setAssignedTo(UUID v) { assignedTo = v; }
    public CaseStatus getStatus() { return status; }
    public void setStatus(CaseStatus v) { status = v; }
    public CasePriority getPriority() { return priority; }
    public void setPriority(CasePriority v) { priority = v; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant v) { updatedAt = v; }
    public Instant getClosedAt() { return closedAt; }
    public void setClosedAt(Instant v) { closedAt = v; }
}
