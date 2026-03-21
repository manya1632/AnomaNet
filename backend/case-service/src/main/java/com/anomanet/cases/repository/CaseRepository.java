package com.anomanet.cases.repository;

import com.anomanet.cases.model.Case;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.UUID;

public interface CaseRepository extends JpaRepository<Case, UUID> {
    Page<Case> findByStatus(Case.CaseStatus status, Pageable pageable);
    Page<Case> findByAssignedTo(UUID assignedTo, Pageable pageable);
}
