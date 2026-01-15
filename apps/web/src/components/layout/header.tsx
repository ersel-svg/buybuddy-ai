"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Sun, Moon, User, LogOut, RefreshCw, XCircle, Loader2 } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { logout, getUser } from "@/lib/auth";
import { toast } from "sonner";

export function Header() {
  const { setTheme } = useTheme();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [mounted, setMounted] = useState(false);
  const [user, setUser] = useState<{ username: string; token: string } | null>(null);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);

  // Only access localStorage after mount to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
    setUser(getUser());
  }, []);

  // Fetch active jobs with smart polling (faster when jobs are running)
  const { data: activeJobsData } = useQuery({
    queryKey: ["jobs", "active", "count"],
    queryFn: () => apiClient.getActiveJobsCount(),
    refetchInterval: (query) => {
      // Poll faster when there are active jobs
      const count = query.state.data?.count ?? 0;
      return count > 0 ? 3000 : 30000; // 3s if active, 30s if idle
    },
  });

  const activeJobsCount = activeJobsData?.count ?? 0;

  // Cancel all jobs mutation
  const cancelAllMutation = useMutation({
    mutationFn: () => apiClient.cancelJobsBatch({}),
    onSuccess: (result) => {
      toast.success(`Cancelled ${result.cancelled_count} jobs`);
      if (result.failed_count > 0) {
        toast.warning(`${result.failed_count} jobs failed to cancel`);
      }
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      setCancelDialogOpen(false);
    },
    onError: () => {
      toast.error("Failed to cancel jobs");
    },
  });

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <header className="h-16 border-b bg-card px-6 flex items-center justify-between">
      {/* Left side - Page title area (can be customized per page) */}
      <div className="flex items-center gap-4">
        {/* Breadcrumbs or title will go here */}
      </div>

      {/* Right side - Actions */}
      <div className="flex items-center gap-2">
        {/* Active Jobs Indicator with Cancel Option */}
        {activeJobsCount > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span className="text-sm">
                  {activeJobsCount} job{activeJobsCount > 1 ? "s" : ""} running
                </span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={() => setCancelDialogOpen(true)}
                className="text-destructive"
              >
                <XCircle className="mr-2 h-4 w-4" />
                Cancel All Jobs
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}

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
                  {mounted && user?.username ? user.username.charAt(0).toUpperCase() : <User className="h-4 w-4" />}
                </AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {mounted && user && (
              <DropdownMenuItem disabled className="text-muted-foreground">
                <User className="mr-2 h-4 w-4" />
                {user.username}
              </DropdownMenuItem>
            )}
            <DropdownMenuItem onClick={handleLogout} className="text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Cancel All Jobs Dialog */}
      <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel All Jobs?</AlertDialogTitle>
            <AlertDialogDescription>
              This will cancel all {activeJobsCount} running/queued jobs and reset their associated
              videos and products to pending status. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep Running</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => cancelAllMutation.mutate()}
              disabled={cancelAllMutation.isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {cancelAllMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <XCircle className="h-4 w-4 mr-2" />
              )}
              Cancel All Jobs
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </header>
  );
}
