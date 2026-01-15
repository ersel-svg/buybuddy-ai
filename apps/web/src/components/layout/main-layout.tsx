"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { type ReactNode } from "react";

interface MainLayoutProps {
  children: ReactNode;
}

// Routes that should not show the sidebar and header
const AUTH_ROUTES = ["/login"];

export function MainLayout({ children }: MainLayoutProps) {
  const pathname = usePathname();

  // Check if current path is an auth route (no sidebar/header)
  const isAuthRoute = AUTH_ROUTES.some((route) => pathname.startsWith(route));

  // Auth routes render without layout chrome
  if (isAuthRoute) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">{children}</main>
      </div>
    </div>
  );
}
