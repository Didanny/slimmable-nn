import os

models = [
    'cifar10_usresnet20',
    'cifar10_usresnet32',
    'cifar10_usresnet44',
    'cifar10_usresnet56',
    'cifar100_usresnet20',
    'cifar100_usresnet32',
    'cifar100_usresnet44',
    'cifar100_usresnet56',
    'cifar10_usvgg_bn11',
    'cifar10_usvgg_bn13',
    'cifar10_usvgg_bn16',
    'cifar10_usvgg_bn19',
    'cifar100_usvgg_bn11',
    'cifar100_usvgg_bn13',
    'cifar100_usvgg_bn16',
    'cifar100_usvgg_bn19',
]

lines = [
    '#!/bin/sh \n' +
    '#SBATCH -n 1 \n' +              # Number of tasks to run (equal to 1 cpu/core per task)
    '#SBATCH -N 1 \n' +              # Ensure that all cores are on one machine
    '#SBATCH -p opengpu.p \n' +      # Partition to submit to
    '#SBATCH --gres=gpu:1 \n' +      # Number of GPU cores requested
    '#SBATCH -o slurm_logs/log_',               # File to which STDOUT will be written, %j inserts jobid
    '#SBATCH -e slurm_logs/err_',
]

def main():
    for model in models:
        dataset = model.split('_')[0]
        with open(f'./slurm/{model}.sh', 'w') as f:
            for line in lines:
                    f.write(f'{line}{model}.out \n')
            f.write(f'python train_cifar.py --model {model} --dataset {dataset} --inplace-distill')
            
    # Run the scripts
    os.system('conda activate torch2')
    for model in models:
        os.system(f'sbatch ./slurm/{model}.sh')

if __name__ == '__main__':
    main()