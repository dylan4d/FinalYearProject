import optuna

# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study_NoDSP_Random'

# Create a new study or load an existing study
pruner = optuna.pruners.PercentilePruner(99)
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True, pruner=pruner)



# After optimization, use he best trial to set the state of policy_net
best_trial = study.best_trial
best_model_path = 'cartpole_v1_best_model_NoDSP_Random.pth'

study = optuna.load_study(study_name=study_name, storage=storage_url)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print(study.best_trial.number)