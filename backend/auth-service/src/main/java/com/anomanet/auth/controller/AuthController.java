package com.anomanet.auth.controller;

import com.anomanet.auth.dto.AuthDtos;
import com.anomanet.auth.service.AuthService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public ResponseEntity<AuthDtos.LoginResponse> login(@RequestBody AuthDtos.LoginRequest request) {
        return ResponseEntity.ok(authService.login(request));
    }

    @PostMapping("/refresh")
    public ResponseEntity<AuthDtos.LoginResponse> refresh(@RequestBody AuthDtos.RefreshRequest request) {
        return ResponseEntity.ok(authService.refresh(request.getRefreshToken()));
    }

    @PostMapping("/logout")
    public ResponseEntity<Map<String, Boolean>> logout() {
        return ResponseEntity.ok(Map.of("success", true));
    }
}
