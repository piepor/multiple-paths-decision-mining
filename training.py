import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from c4dot5.DecisionTreeClassifier import DecisionTreeClassifier



def sampling_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """ Performs sampling to obtain a balanced dataset, in terms of target values. """

    dataset = dataset.copy()

    groups = list()
    grouped_df = dataset.groupby('target')
    for target_value in dataset['target'].unique():
        groups.append(grouped_df.get_group(target_value))
    groups.sort(key=len)
    # Groups is a list containing a dataset for each target value, ordered by length
    # If the smaller datasets are less than the 35% of the total dataset length, then apply the sampling
    if sum(len(group) for group in groups[:-1]) / len(dataset) <= 0.35:
        samples = list()
        # Each smaller dataset is appended to the 'samples' list, along with a sampled dataset from the largest one
        for group in groups[:-1]:
            samples.append(group)
            samples.append(groups[-1].sample(len(group)))
        # The datasets in the 'samples' list are then concatenated together
        dataset = pd.concat(samples, ignore_index=True)

    return dataset

def train(training_data: dict, attributes_map: dict, n_sample: int=10):
    #file_name = 'test.txt'
    for decision_point in training_data.keys():
        #if decision_point == 'p_3':
        print("\nDecision point: {}".format(decision_point))
        complete_dataset = pd.DataFrame.from_dict(training_data[decision_point])
        # Replacing ':' with '_' both in the dataset columns and in the attributes map since ':' creates problems
        #breakpoint()
        complete_dataset.columns = complete_dataset.columns.str.replace(':', '_')
        attributes_map = {k.replace(':', '_'): attributes_map[k] for k in attributes_map}

        print("Fitting a decision tree on the decision point's dataset...")
        accuracies, f_scores = list(), list()
        for i in tqdm(range(n_sample)):
            # Sampling
            #dataset = sampling_dataset(complete_dataset)
            dataset = complete_dataset.copy()

            # Fitting
            dt = DecisionTreeClassifier(attributes_map.copy())
            dt.fit(dataset)
            dt.view(title=f"{decision_point}-{i+1}")

            # Predict
            y_pred = dt.predict(dataset.drop(columns=['target']))

            # Accuracy
            accuracy = metrics.accuracy_score(dataset['target'], y_pred)
            accuracies.append(accuracy)

            # F1-score
            if len(dataset['target'].unique()) > 2:
                f1_score = metrics.f1_score(dataset['target'], y_pred, average='weighted')
            else:
                f1_score = metrics.f1_score(dataset['target'], y_pred, pos_label=dataset['target'].unique()[0])
            f_scores.append(f1_score)
            # get rules
            rules = dt.get_rules('standard')
            for rule in rules:
                print(f"{rule}: \n {rules[rule]}")
        print("Train accuracy: {}".format(sum(accuracies) / len(accuracies)))
        print("F1 score: {}".format(sum(f_scores) / len(f_scores)))

