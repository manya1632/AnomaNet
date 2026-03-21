package com.anomanet.cases.controller;

import com.anomanet.cases.model.Case;
import com.anomanet.cases.model.CaseNote;
import com.anomanet.cases.repository.CaseNoteRepository;
import com.anomanet.cases.repository.CaseRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/cases")
public class CaseController {

    private final CaseRepository caseRepo;
    private final CaseNoteRepository noteRepo;

    public CaseController(CaseRepository caseRepo, CaseNoteRepository noteRepo) {
        this.caseRepo = caseRepo;
        this.noteRepo = noteRepo;
    }

    @GetMapping
    public ResponseEntity<Page<Case>> list(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(caseRepo.findAll(
                PageRequest.of(page, size, Sort.by("createdAt").descending())));
    }

    @PostMapping
    public ResponseEntity<Case> create(@RequestBody Map<String, String> body) {
        Case c = new Case();
        c.setAlertId(UUID.fromString(body.get("alertId")));
        if (body.containsKey("assignedTo"))
            c.setAssignedTo(UUID.fromString(body.get("assignedTo")));
        if (body.containsKey("priority"))
            c.setPriority(Case.CasePriority.valueOf(body.get("priority")));
        return ResponseEntity.ok(caseRepo.save(c));
    }

    @GetMapping("/{id}")
    public ResponseEntity<Case> getById(@PathVariable UUID id) {
        return caseRepo.findById(id).map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}/status")
    public ResponseEntity<Case> updateStatus(@PathVariable UUID id,
                                              @RequestBody Map<String, String> body) {
        return caseRepo.findById(id).map(c -> {
            c.setStatus(Case.CaseStatus.valueOf(body.get("status")));
            c.setUpdatedAt(Instant.now());
            if (c.getStatus() == Case.CaseStatus.CLOSED_SAR ||
                c.getStatus() == Case.CaseStatus.CLOSED_FP)
                c.setClosedAt(Instant.now());
            return ResponseEntity.ok(caseRepo.save(c));
        }).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/{id}/notes")
    public ResponseEntity<List<CaseNote>> getNotes(@PathVariable UUID id) {
        return ResponseEntity.ok(noteRepo.findByCaseIdOrderByCreatedAtAsc(id));
    }

    @PostMapping("/{id}/notes")
    public ResponseEntity<CaseNote> addNote(@PathVariable UUID id,
                                             @RequestBody Map<String, String> body,
                                             @RequestHeader(value = "X-User-Name", defaultValue = "system") String user) {
        return caseRepo.findById(id).map(c -> {
            CaseNote note = new CaseNote();
            note.setCaseId(id);
            note.setAuthorId(UUID.nameUUIDFromBytes(user.getBytes()));
            note.setNote(body.get("note"));
            return ResponseEntity.ok(noteRepo.save(note));
        }).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Long>> stats() {
        return ResponseEntity.ok(Map.of(
            "total", caseRepo.count(),
            "open", caseRepo.findByStatus(Case.CaseStatus.OPEN,
                PageRequest.of(0, 1)).getTotalElements(),
            "escalated", caseRepo.findByStatus(Case.CaseStatus.ESCALATED,
                PageRequest.of(0, 1)).getTotalElements()
        ));
    }
}
