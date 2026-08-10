import numpy as np

from simulator import (
    BusinessInputs,
    UncertaintyInputs,
    calculate_metrics,
    run_monte_carlo,
    scenario_summary,
)


def test_base_case_calculations():
    inputs = BusinessInputs(
        price=20,
        customers=100,
        units_per_customer=2,
        rent=500,
        employees=1,
        salary_per_employee=500,
        material_cost_per_unit=5,
        marketing_budget=100,
        other_fixed_costs=100,
        seasonal_demand_percent=10,
        tax_rate_percent=20,
        profit_target=800,
    )
    result = calculate_metrics(inputs)

    assert np.isclose(result["customers"], 110)
    assert np.isclose(result["units"], 220)
    assert np.isclose(result["revenue"], 4_400)
    assert np.isclose(result["material_costs"], 1_100)
    assert np.isclose(result["fixed_costs"], 1_200)
    assert np.isclose(result["profit_before_tax"], 2_100)
    assert np.isclose(result["taxes"], 420)
    assert np.isclose(result["net_profit"], 1_680)
    assert np.isclose(result["break_even_customers"], 40)
    assert np.isclose(result["required_customers"], 73.3333333333)


def test_losses_are_not_given_negative_tax():
    result = calculate_metrics(BusinessInputs(customers=0, tax_rate_percent=30))
    assert result["profit_before_tax"] < 0
    assert result["taxes"] == 0
    assert result["net_profit"] == result["profit_before_tax"]


def test_non_positive_contribution_has_no_break_even():
    result = calculate_metrics(BusinessInputs(price=5, material_cost_per_unit=5))
    assert np.isinf(result["break_even_units"])
    assert np.isinf(result["required_customers"])


def test_zero_uncertainty_matches_base_case():
    inputs = BusinessInputs()
    base = calculate_metrics(inputs)
    simulations = run_monte_carlo(
        inputs,
        UncertaintyInputs(
            demand_percent=0,
            price_percent=0,
            material_cost_percent=0,
            simulations=100,
            seed=1,
        ),
    )
    assert np.allclose(simulations["Revenue"], base["revenue"])
    assert np.allclose(simulations["Net profit"], base["net_profit"])


def test_simulation_is_reproducible_and_summary_is_ordered():
    inputs = BusinessInputs()
    uncertainty = UncertaintyInputs(simulations=500, seed=123)
    first = run_monte_carlo(inputs, uncertainty)
    second = run_monte_carlo(inputs, uncertainty)
    assert first.equals(second)

    summary = scenario_summary(first)
    assert summary.loc["Worst case (P10)", "Net profit"] <= summary.loc[
        "Expected (P50)", "Net profit"
    ]
    assert summary.loc["Expected (P50)", "Net profit"] <= summary.loc[
        "Best case (P90)", "Net profit"
    ]
