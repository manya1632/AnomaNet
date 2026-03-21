package com.anomanet.auth.dto;

import java.util.UUID;

public class AuthDtos {

    public static class LoginRequest {
        private String username;
        private String password;
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    public static class RefreshRequest {
        private String refreshToken;
        public String getRefreshToken() { return refreshToken; }
        public void setRefreshToken(String refreshToken) { this.refreshToken = refreshToken; }
    }

    public static class LoginResponse {
        private String token;
        private String refreshToken;
        private UserInfo user;

        public LoginResponse() {}
        public LoginResponse(String token, String refreshToken, UserInfo user) {
            this.token = token;
            this.refreshToken = refreshToken;
            this.user = user;
        }
        public static Builder builder() { return new Builder(); }
        public String getToken() { return token; }
        public String getRefreshToken() { return refreshToken; }
        public UserInfo getUser() { return user; }

        public static class Builder {
            private String token;
            private String refreshToken;
            private UserInfo user;
            public Builder token(String token) { this.token = token; return this; }
            public Builder refreshToken(String refreshToken) { this.refreshToken = refreshToken; return this; }
            public Builder user(UserInfo user) { this.user = user; return this; }
            public LoginResponse build() { return new LoginResponse(token, refreshToken, user); }
        }
    }

    public static class UserInfo {
        private UUID id;
        private String name;
        private String role;
        private String username;

        public UserInfo() {}
        public UserInfo(UUID id, String name, String role, String username) {
            this.id = id; this.name = name; this.role = role; this.username = username;
        }
        public static Builder builder() { return new Builder(); }
        public UUID getId() { return id; }
        public String getName() { return name; }
        public String getRole() { return role; }
        public String getUsername() { return username; }

        public static class Builder {
            private UUID id;
            private String name;
            private String role;
            private String username;
            public Builder id(UUID id) { this.id = id; return this; }
            public Builder name(String name) { this.name = name; return this; }
            public Builder role(String role) { this.role = role; return this; }
            public Builder username(String username) { this.username = username; return this; }
            public UserInfo build() { return new UserInfo(id, name, role, username); }
        }
    }
}
