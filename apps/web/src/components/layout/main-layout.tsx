"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { type ReactNode } from "react";
import { getToken } from "@/lib/auth";

interface MainLayoutProps {
  children: ReactNode;
}

// Routes that should not show the sidebar and header
const AUTH_ROUTES = ["/login"];

// Routes that need full-screen layout (no padding, no scroll)
const FULLSCREEN_ROUTES = ["/od/annotate", "/store-maps/editor"];

export function MainLayout({ children }: MainLayoutProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);

  // Check if current path is an auth route (no sidebar/header)
  const isAuthRoute = AUTH_ROUTES.some((route) => pathname.startsWith(route));

  // Check if current path needs fullscreen layout
  const isFullscreenRoute = FULLSCREEN_ROUTES.some((route) => pathname.startsWith(route));

  useEffect(() => {
    // Skip auth check for auth routes
    if (isAuthRoute) {
      setIsChecking(false);
      return;
    }

    const token = getToken();
    if (!token) {
      router.replace(`/login?redirect=${encodeURIComponent(pathname)}`);
    } else {
      setIsChecking(false);
    }
  }, [pathname, isAuthRoute, router]);

  // Auth routes render without layout chrome
  if (isAuthRoute) {
    return <>{children}</>;
  }

  // Show nothing while checking auth (prevents flash of content)
  if (isChecking) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className={`flex-1 ${isFullscreenRoute ? "overflow-hidden" : "overflow-auto p-6"}`}>
          {children}
        </main>
      </div>
    </div>
  );
}
