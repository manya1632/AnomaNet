package com.anomanet.transaction.controller;

import com.anomanet.transaction.model.Transaction;
import com.anomanet.transaction.repository.TransactionRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/transactions")
public class TransactionController {

    private final TransactionRepository repo;

    public TransactionController(TransactionRepository repo) {
        this.repo = repo;
    }

    @GetMapping
    public ResponseEntity<Page<Transaction>> list(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String accountId) {

        PageRequest pageable = PageRequest.of(page, size, Sort.by("initiatedAt").descending());
        if (accountId != null) {
            return ResponseEntity.ok(repo.findBySourceAccountIdOrDestAccountId(accountId, accountId, pageable));
        }
        return ResponseEntity.ok(repo.findAll(pageable));
    }

    @GetMapping("/{id}")
    public ResponseEntity<Transaction> getById(@PathVariable UUID id) {
        return repo.findById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/{id}/trail")
    public ResponseEntity<List<Transaction>> getTrail(@PathVariable UUID id) {
        return repo.findById(id)
                .map(tx -> repo.findTrailByAccountId(tx.getSourceAccountId()))
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
}