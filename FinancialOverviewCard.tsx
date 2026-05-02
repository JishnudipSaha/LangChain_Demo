type FinancialOverviewCardProps = {
  balance?: string;
  trend?: string;
};

export default function FinancialOverviewCard({
  balance = "$24,680.00",
  trend = "+12%",
}: FinancialOverviewCardProps) {
  return (
    <div className="w-full max-w-sm rounded-xl border border-border/60 bg-card p-5 text-card-foreground shadow-sm">
      <p className="text-sm font-medium text-muted-foreground">Financial Overview</p>
      <div className="mt-3 flex items-end justify-between gap-3">
        <div>
          <p className="text-xs text-muted-foreground">Total Balance</p>
          <p className="mt-1 text-3xl font-semibold tracking-tight">{balance}</p>
        </div>
        <span className="inline-flex items-center rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2 py-1 text-xs font-medium text-emerald-600">
          {trend}
        </span>
      </div>
    </div>
  );
}
