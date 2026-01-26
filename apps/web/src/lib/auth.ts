/**
 * Authentication utilities for BuyBuddy AI Platform.
 *
 * Token is stored in localStorage and included in all API requests.
 */

const TOKEN_KEY = "buybuddy_auth_token";
const USER_KEY = "buybuddy_auth_user";

export interface AuthUser {
  username: string;
  token: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  username: string;
  message: string;
}

/**
 * Get the stored auth token.
 */
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * Get the stored user info.
 */
export function getUser(): AuthUser | null {
  if (typeof window === "undefined") return null;

  const token = localStorage.getItem(TOKEN_KEY);
  const username = localStorage.getItem(USER_KEY);

  if (!token || !username) return null;

  return { token, username };
}

/**
 * Store auth credentials after login.
 * Saves to both localStorage (for client) and cookie (for middleware).
 */
export function setAuth(token: string, username: string): void {
  if (typeof window === "undefined") return;

  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username);

  // Set cookie for middleware (httpOnly not possible from client, but still useful for SSR)
  document.cookie = `buybuddy_auth_token=${token}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;
}

/**
 * Clear auth credentials (logout).
 */
export function clearAuth(): void {
  if (typeof window === "undefined") return;

  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);

  // Clear cookie
  document.cookie = "buybuddy_auth_token=; path=/; max-age=0";
}

/**
 * Check if user is authenticated.
 */
export function isAuthenticated(): boolean {
  return getToken() !== null;
}

/**
 * Login with credentials.
 */
export async function login(credentials: LoginCredentials): Promise<AuthUser> {
  const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Login failed");
  }

  const data: LoginResponse = await response.json();

  // Store credentials
  setAuth(data.token, data.username);

  return {
    token: data.token,
    username: data.username,
  };
}

/**
 * Logout and clear credentials.
 */
export async function logout(): Promise<void> {
  const token = getToken();

  if (token) {
    const API_BASE_URL =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // Try to invalidate token on server (best effort)
    try {
      await fetch(`${API_BASE_URL}/api/v1/auth/logout`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });
    } catch {
      // Ignore errors - we'll clear local storage anyway
    }
  }

  clearAuth();
}

/**
 * Get authorization header for API requests.
 */
export function getAuthHeader(): Record<string, string> {
  const token = getToken();
  if (!token) return {};

  return {
    Authorization: `Bearer ${token}`,
  };
}
