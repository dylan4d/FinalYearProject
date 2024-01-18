import optuna

# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study_NoDSP'

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)

# Print the number of finished trials
print("Number of finished trials: ", len(study.trials))

# Print information about the best trial
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Trial Number: " ,trial.number)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
