package com.anomanet.auth.service;

import com.anomanet.auth.dto.AuthDtos;
import com.anomanet.auth.model.User;
import com.anomanet.auth.repository.UserRepository;
import com.anomanet.auth.security.JwtUtil;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService {

    private final UserRepository userRepository;
    private final JwtUtil jwtUtil;
    private final PasswordEncoder passwordEncoder;

    public AuthService(UserRepository userRepository, JwtUtil jwtUtil, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.jwtUtil = jwtUtil;
        this.passwordEncoder = passwordEncoder;
    }

    public AuthDtos.LoginResponse login(AuthDtos.LoginRequest request) {
        User user = userRepository.findByUsername(request.getUsername())
                .orElseThrow(() -> new RuntimeException("User not found"));
        if (!user.isActive())
            throw new RuntimeException("Account disabled");
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash()))
            throw new RuntimeException("Invalid credentials");
        return buildResponse(user);
    }

    public AuthDtos.LoginResponse refresh(String refreshToken) {
        if (!jwtUtil.isValid(refreshToken))
            throw new RuntimeException("Invalid refresh token");
        String username = jwtUtil.extractUsername(refreshToken);
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return buildResponse(user);
    }

    private AuthDtos.LoginResponse buildResponse(User user) {
        return AuthDtos.LoginResponse.builder()
                .token(jwtUtil.generateToken(user.getUsername(), user.getRole().name()))
                .refreshToken(jwtUtil.generateRefreshToken(user.getUsername()))
                .user(AuthDtos.UserInfo.builder()
                        .id(user.getId())
                        .name(user.getFullName())
                        .role(user.getRole().name())
                        .username(user.getUsername())
                        .build())
                .build();
    }
}