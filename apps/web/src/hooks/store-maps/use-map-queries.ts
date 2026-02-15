import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

// ===========================================
// Query Keys
// ===========================================

export const storeMapKeys = {
  all: ["store-maps"] as const,
  lists: () => [...storeMapKeys.all, "list"] as const,
  list: () => [...storeMapKeys.lists()] as const,
  details: () => [...storeMapKeys.all, "detail"] as const,
  detail: (id: number) => [...storeMapKeys.details(), id] as const,
  floors: (mapId: number) => [...storeMapKeys.all, "floors", mapId] as const,
  coordinates: (filters: Record<string, unknown>) =>
    [...storeMapKeys.all, "coordinates", filters] as const,
};

// ===========================================
// Map Queries
// ===========================================

export function useStoreMaps() {
  return useQuery({
    queryKey: storeMapKeys.list(),
    queryFn: async () => {
      const response = await apiClient.getStoreMaps();
      // BuyBuddy API wraps arrays in { data: [...] }
      if (response && !Array.isArray(response) && Array.isArray((response as any).data)) {
        return (response as any).data;
      }
      return response;
    },
  });
}

export function useStoreMap(id: number) {
  return useQuery({
    queryKey: storeMapKeys.detail(id),
    queryFn: () => apiClient.getStoreMap(id),
    enabled: id > 0,
  });
}

export function useCreateStoreMap() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      store_id: number;
      name: string;
      base_ratio?: number;
      grid?: number;
    }) => apiClient.createStoreMap(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.lists() });
      toast.success("Map created successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to create map: ${error.message}`);
    },
  });
}

export function useUpdateStoreMap() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      ...data
    }: {
      id: number;
      name?: string;
      base_ratio?: number;
      grid?: number;
    }) => apiClient.updateStoreMap(id, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: storeMapKeys.detail(variables.id),
      });
      queryClient.invalidateQueries({ queryKey: storeMapKeys.lists() });
    },
    onError: (error: Error) => {
      toast.error(`Failed to update map: ${error.message}`);
    },
  });
}

export function useDeleteStoreMap() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteStoreMap(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.lists() });
      toast.success("Map deleted successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete map: ${error.message}`);
    },
  });
}

// ===========================================
// Floor Queries
// ===========================================

export function useMapFloors(mapId: number) {
  return useQuery({
    queryKey: storeMapKeys.floors(mapId),
    queryFn: async () => {
      const response = await apiClient.getMapFloors(mapId);
      if (response && !Array.isArray(response) && Array.isArray((response as any).data)) {
        return (response as any).data;
      }
      return response;
    },
    enabled: mapId > 0,
  });
}

export function useCreateMapFloor() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      mapId,
      floor,
      file,
    }: {
      mapId: number;
      floor: number;
      file: File;
    }) => apiClient.createMapFloor(mapId, floor, file),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: storeMapKeys.floors(variables.mapId),
      });
      toast.success("Floor added successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to add floor: ${error.message}`);
    },
  });
}

// ===========================================
// Area Queries
// ===========================================

export function useCreateMapArea() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { name: string; floor_id: number }) =>
      apiClient.createMapArea(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
    },
    onError: (error: Error) => {
      toast.error(`Failed to create area: ${error.message}`);
    },
  });
}

export function useUpdateMapArea() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { id: number; name?: string }) =>
      apiClient.updateMapArea(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
    },
    onError: (error: Error) => {
      toast.error(`Failed to update area: ${error.message}`);
    },
  });
}

export function useDeleteMapArea() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteMapArea(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
      toast.success("Area deleted");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete area: ${error.message}`);
    },
  });
}

// ===========================================
// Coordinate Queries
// ===========================================

export function useAreaCoordinates(filters: {
  map_id?: number;
  store_id?: number;
  area_id?: number;
  floor_id?: number;
}) {
  return useQuery({
    queryKey: storeMapKeys.coordinates(filters),
    queryFn: async () => {
      const response = await apiClient.getAreaCoordinates(filters);
      if (response && !Array.isArray(response) && Array.isArray((response as any).data)) {
        return (response as any).data;
      }
      return response;
    },
    enabled: !!(filters.map_id || filters.store_id),
  });
}

export function useCreateAreaCoordinate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      area_id: number;
      x: number;
      y: number;
      z?: number;
      r?: number;
      circle?: boolean;
    }) => apiClient.createAreaCoordinate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
    },
    onError: (error: Error) => {
      toast.error(`Failed to create coordinate: ${error.message}`);
    },
  });
}

export function useUpdateAreaCoordinate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      id: number;
      x?: number;
      y?: number;
      z?: number;
      r?: number;
    }) => apiClient.updateAreaCoordinate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
    },
  });
}

export function useDeleteAreaCoordinate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteAreaCoordinate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: storeMapKeys.all });
    },
  });
}
