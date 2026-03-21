package com.anomanet.transaction.repository;

import com.anomanet.transaction.model.Transaction;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

public interface TransactionRepository extends JpaRepository<Transaction, UUID> {

    Page<Transaction> findBySourceAccountIdOrDestAccountId(
            String sourceAccountId, String destAccountId, Pageable pageable);

    @Query("SELECT t FROM Transaction t WHERE " +
           "t.sourceAccountId = :accountId OR t.destAccountId = :accountId " +
           "ORDER BY t.initiatedAt ASC")
    List<Transaction> findTrailByAccountId(@Param("accountId") String accountId);
}