package com.anomanet.cases.repository;

import com.anomanet.cases.model.CaseNote;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface CaseNoteRepository extends JpaRepository<CaseNote, UUID> {
    List<CaseNote> findByCaseIdOrderByCreatedAtAsc(UUID caseId);
}
