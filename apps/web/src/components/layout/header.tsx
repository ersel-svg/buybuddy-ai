"use client";

import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Sun, Moon, Bell, User, LogOut, RefreshCw } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

export function Header() {
  const { theme, setTheme } = useTheme();

  // Fetch active jobs count for notification badge
  const { data: jobs } = useQuery({
    queryKey: ["jobs", "active"],
    queryFn: () => apiClient.getJobs(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const activeJobsCount = jobs?.filter(
    (job) => job.status === "running" || job.status === "queued"
  ).length;

  return (
    <header className="h-16 border-b bg-card px-6 flex items-center justify-between">
      {/* Left side - Page title area (can be customized per page) */}
      <div className="flex items-center gap-4">
        {/* Breadcrumbs or title will go here */}
      </div>

      {/* Right side - Actions */}
      <div className="flex items-center gap-2">
        {/* Active Jobs Indicator */}
        {activeJobsCount && activeJobsCount > 0 ? (
          <Button variant="ghost" size="sm" className="gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span className="text-sm">
              {activeJobsCount} job{activeJobsCount > 1 ? "s" : ""} running
            </span>
          </Button>
        ) : null}

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="relative">
          <Bell className="w-5 h-5" />
          {activeJobsCount && activeJobsCount > 0 && (
            <Badge
              variant="destructive"
              className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs"
            >
              {activeJobsCount}
            </Badge>
          )}
        </Button>

        {/* Theme Toggle */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => setTheme("light")}>
              <Sun className="mr-2 h-4 w-4" />
              Light
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTheme("dark")}>
              <Moon className="mr-2 h-4 w-4" />
              Dark
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setTheme("system")}>
              <span className="mr-2">💻</span>
              System
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="rounded-full">
              <Avatar className="h-8 w-8">
                <AvatarFallback>
                  <User className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem>
              <User className="mr-2 h-4 w-4" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem className="text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
