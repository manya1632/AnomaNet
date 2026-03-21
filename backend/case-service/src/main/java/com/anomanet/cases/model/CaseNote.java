package com.anomanet.cases.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "case_notes")
public class CaseNote {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "case_id", nullable = false)
    private UUID caseId;

    @Column(name = "author_id", nullable = false)
    private UUID authorId;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String note;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();

    public UUID getId() { return id; }
    public UUID getCaseId() { return caseId; }
    public void setCaseId(UUID v) { caseId = v; }
    public UUID getAuthorId() { return authorId; }
    public void setAuthorId(UUID v) { authorId = v; }
    public String getNote() { return note; }
    public void setNote(String v) { note = v; }
    public Instant getCreatedAt() { return createdAt; }
}
