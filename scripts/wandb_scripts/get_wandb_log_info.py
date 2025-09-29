import argparse
import wandb

def update_config(run):
    run.config['key'] = updated_value
    run.update()

def export_as_csv(run):
    metrics_dataframe = run.history()
    metrics_dataframe.to_csv('1000ep_metrics.csv')

def read_metrics(run):
    if run.state == 'finished':
        for i, row in run.history().iterrows():
            print(row['_timestamp'], row['accuracy'])

def get_unsampled_metric_data(run):
    history = run.scan_history()
    losses = [row['loss'] for row in history]
    return losses

def best_model_from_sweep(sweepid):
    runs = sorted(sweep, runs,
                  key = lambda run: run.summary.get('val_acc', 0),
                  reverse = True)
    val_acc = runs[0].summary.get('vall_acc', 0)
    print(' '.join(f'Best run {runs[0].name} with',
                f'{vall_acc} % validation accuracy'))
    runs[0].file('model.h5').download(replace=True)
    print('Best model saved to model-best.h5')

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity',
                        help = 'Entity for WandB',
                        type = str,
                        default = 'aravindnair-98-iit-delhi')
    parser.add_argument('--project',
                        help = 'Project name for WandB',
                        type = str,
                        default = 'MPMorph_Li')
    parser.add_argument('--sweepid',
                        help = 'Sweep ID',
                        type = str,
                        default = None)
    parser.add_argument('--runid',
                        help = 'Run ID',
                        type = str,
                        default = '2025-09-03-14-59-36')
    return parser

if __name__ == '__main__':
    api = wandb.Api()
    parser = create_arg_parser()
    args = parser.parse_args()
    entity = args.entity
    project = args.project
    runid = args.runid
    sweepid = args.sweepid

    assert runid
    run = api.run('/'.join([entity, project, runid]))
    export_as_csv(run) 
