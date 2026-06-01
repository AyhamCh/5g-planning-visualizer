import { queryOptions } from "@tanstack/react-query";
import { fetchGrid, fetchSites } from "./telco-data";

export const gridQuery = queryOptions({
  queryKey: ["grid"],
  queryFn: fetchGrid,
  staleTime: 60_000,
});

export const sitesQuery = queryOptions({
  queryKey: ["sites"],
  queryFn: fetchSites,
  staleTime: 60_000,
});
