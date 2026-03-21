package com.anomanet.graph.controller;

import com.anomanet.graph.dto.GraphDtos;
import com.anomanet.graph.service.GraphQueryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/graph")
public class GraphController {

    private final GraphQueryService graphQueryService;

    public GraphController(GraphQueryService graphQueryService) {
        this.graphQueryService = graphQueryService;
    }

    @PostMapping("/subgraph")
    public ResponseEntity<GraphDtos.SubgraphResponse> subgraph(
            @RequestBody GraphDtos.SubgraphRequest req) {
        return ResponseEntity.ok(graphQueryService.getSubgraph(
                req.getAccountId(),
                req.getDepth(),
                req.getHours()));
    }

    @PostMapping("/cycles")
    public ResponseEntity<List<GraphDtos.CycleResult>> cycles(
            @RequestBody GraphDtos.CycleRequest req) {
        return ResponseEntity.ok(graphQueryService.detectCycles(
                req.getAccountId(),
                req.getMaxLength(),
                req.getHours()));
    }

    @GetMapping("/account/{id}/stats")
    public ResponseEntity<GraphDtos.AccountStats> stats(@PathVariable String id) {
        return ResponseEntity.ok(graphQueryService.getAccountStats(id));
    }
}
