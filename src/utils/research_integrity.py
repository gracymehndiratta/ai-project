"""Conservative safeguards for historical experiments and result presentation."""
import warnings


def contaminated(model):
    info = model.get('info', model)
    return bool(model.get('stage') == 'adapted'
                or info.get('data_contaminated_for_test_evaluation')
                or info.get('excluded_from_clean_model_ranking')
                or info.get('training_variant') in {'joint_refinement', 'test_adaptation', 'six_model_adaptation'}
                or any('adaptation' in part or 'joint_refinement' in part
                       for part in str(model.get('folder', '')).replace('\\', '/').split('/')))


def presentation_order(models):
    """Never select or order a checkpoint using inspected evaluation scores."""
    return sorted(models, key=lambda model: (model.get('name', '').casefold(), model['id']))


def require_exploratory_opt_in(allow_test_fitting=False):
    if not allow_test_fitting:
        raise ValueError('Former-test fitting is disabled. Use a train-only pipeline. '
                         'Historical reproduction requires --allow-test-fitting; '
                         'its outputs are fitting results, never independent test evidence.')
    warnings.warn('Historical former-test fitting enabled: outputs cannot establish '
                  'generalization or superiority over an unadapted reference.', UserWarning)
