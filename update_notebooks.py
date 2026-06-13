import glob

for f in glob.glob('notebooks/*.ipynb'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    content = content.replace('wc2018_master_dataset.csv', 'wc2026_master_dataset_updated.csv')
    content = content.replace('team_historical_features.csv', 'team_historical_features_updated.csv')
    content = content.replace('wc_training_dataset.csv', 'wc_training_dataset_updated.csv')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print('Successfully updated notebooks!')
