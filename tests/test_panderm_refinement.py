from src.utils.panderm_refinement import F1Stopping


def test_stopping_tracks_f1_even_if_logloss_is_also_present():
    class Booster:
        def set_attr(self,**values): self.values=values
    model=Booster()
    callback=F1Stopping()
    for epoch in range(400):
        # Log loss improves continuously while F1 peaked at round 10.
        history={'validation_0':{'mlogloss':[1/(epoch+1)],
            'macro_f1_cost':[-.8 if epoch==9 else -.5]}}
        stopped=callback.after_iteration(model,epoch,history)
        assert stopped==(epoch==399)
    assert model.values['best_iteration']=='9'
