"""Financial model and Monte Carlo engine for the Tunisia Profit Simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessInputs:
    """Deterministic monthly business assumptions."""

    price: float = 25.0
    customers: int = 450
    units_per_customer: float = 1.2
    rent: float = 1_500.0
    employees: int = 2
    salary_per_employee: float = 900.0
    material_cost_per_unit: float = 8.0
    marketing_budget: float = 600.0
    other_fixed_costs: float = 350.0
    seasonal_demand_percent: float = 0.0
    tax_rate_percent: float = 15.0
    profit_target: float = 3_000.0


@dataclass(frozen=True)
class UncertaintyInputs:
    """Symmetric ranges used to sample uncertain assumptions."""

    demand_percent: float = 20.0
    price_percent: float = 5.0
    material_cost_percent: float = 12.0
    simulations: int = 5_000
    seed: int = 42


def calculate_metrics(inputs: BusinessInputs) -> dict[str, float]:
    """Calculate the base monthly profit-and-loss metrics."""
    seasonal_multiplier = max(0.0, 1.0 + inputs.seasonal_demand_percent / 100)
    customers = inputs.customers * seasonal_multiplier
    units = customers * inputs.units_per_customer
    revenue = units * inputs.price
    material_costs = units * inputs.material_cost_per_unit
    gross_profit = revenue - material_costs
    salaries = inputs.employees * inputs.salary_per_employee
    fixed_costs = inputs.rent + salaries + inputs.marketing_budget + inputs.other_fixed_costs
    profit_before_tax = gross_profit - fixed_costs
    taxes = max(0.0, profit_before_tax) * inputs.tax_rate_percent / 100
    net_profit = profit_before_tax - taxes
    contribution_per_unit = inputs.price - inputs.material_cost_per_unit

    if contribution_per_unit > 0:
        break_even_units = fixed_costs / contribution_per_unit
        break_even_customers = break_even_units / inputs.units_per_customer
        target_before_tax = (
            inputs.profit_target / (1 - inputs.tax_rate_percent / 100)
            if inputs.profit_target > 0 and inputs.tax_rate_percent < 100
            else inputs.profit_target
        )
        required_units = (fixed_costs + target_before_tax) / contribution_per_unit
        required_customers = required_units / inputs.units_per_customer
    else:
        break_even_units = np.inf
        break_even_customers = np.inf
        required_units = np.inf
        required_customers = np.inf

    return {
        "customers": customers,
        "units": units,
        "revenue": revenue,
        "material_costs": material_costs,
        "gross_profit": gross_profit,
        "gross_margin_percent": gross_profit / revenue * 100 if revenue else 0.0,
        "salaries": salaries,
        "fixed_costs": fixed_costs,
        "profit_before_tax": profit_before_tax,
        "taxes": taxes,
        "net_profit": net_profit,
        "monthly_cash_flow": net_profit,
        "contribution_per_unit": contribution_per_unit,
        "break_even_units": break_even_units,
        "break_even_customers": break_even_customers,
        "required_units": required_units,
        "required_customers": required_customers,
    }


def run_monte_carlo(
    inputs: BusinessInputs, uncertainty: UncertaintyInputs
) -> pd.DataFrame:
    """Simulate monthly outcomes using triangular distributions.

    Each range is centered on the user's base assumption. Triangular sampling is
    intentionally used because the base value is assumed to be more likely than
    either extreme while remaining easy for a non-statistician to understand.
    """
    rng = np.random.default_rng(uncertainty.seed)
    n = uncertainty.simulations

    def sample_multiplier(percent: float) -> np.ndarray:
        width = max(0.0, percent) / 100
        if width == 0:
            return np.ones(n)
        return rng.triangular(1 - width, 1.0, 1 + width, n)

    seasonal_multiplier = max(0.0, 1 + inputs.seasonal_demand_percent / 100)
    customers = inputs.customers * seasonal_multiplier * sample_multiplier(
        uncertainty.demand_percent
    )
    units = customers * inputs.units_per_customer
    prices = inputs.price * sample_multiplier(uncertainty.price_percent)
    material_unit_costs = inputs.material_cost_per_unit * sample_multiplier(
        uncertainty.material_cost_percent
    )
    revenue = units * prices
    material_costs = units * material_unit_costs
    gross_profit = revenue - material_costs
    fixed_costs = (
        inputs.rent
        + inputs.employees * inputs.salary_per_employee
        + inputs.marketing_budget
        + inputs.other_fixed_costs
    )
    profit_before_tax = gross_profit - fixed_costs
    taxes = np.maximum(profit_before_tax, 0) * inputs.tax_rate_percent / 100
    net_profit = profit_before_tax - taxes

    return pd.DataFrame(
        {
            "Customers": customers,
            "Units sold": units,
            "Price": prices,
            "Material cost / unit": material_unit_costs,
            "Revenue": revenue,
            "Gross profit": gross_profit,
            "Net profit": net_profit,
        }
    )


def scenario_summary(simulations: pd.DataFrame) -> pd.DataFrame:
    """Return easy-to-read downside, expected, and upside percentiles."""
    percentiles = simulations[["Revenue", "Net profit", "Customers"]].quantile(
        [0.10, 0.50, 0.90]
    )
    percentiles.index = ["Worst case (P10)", "Expected (P50)", "Best case (P90)"]
    return percentiles


def inputs_as_dict(inputs: BusinessInputs) -> dict[str, float]:
    """Expose dataclass values for exports without coupling UI code to dataclasses."""
    return asdict(inputs)
