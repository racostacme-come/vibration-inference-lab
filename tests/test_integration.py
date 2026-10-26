from integration import experiment

def test_shared_model_contract():
    result = experiment()
    assert result["transition_max_difference"] < 1e-11
    assert result["max_smoothed_minus_filtered_variance"] < 1e-12
    assert result["missing_samples"] == 51
    # A single realization need not improve; test the second-moment guarantee.
    ensemble = result['ensemble']
    assert ensemble['smoothed_expected_mse'] < ensemble['filtered_expected_mse']
    for method in ('filtered', 'smoothed'):
        assert abs(ensemble[method+'_mse']-ensemble[method+'_expected_mse']) < 6*ensemble[method+'_mse_standard_error']
