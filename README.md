# autonomous-car-model
Deep learning model that mimics human driver behavior and drives simulated car using Udacity self-driving simulator.

## Installation

Install virtual environmenet and package manager
```bash
pip install pipenv
```

Install required packages
```bash
pipenv install
```

Activate virtual environment
```bash
pipenv shell
```

## Data

Udacity is providing us with a set of data generated using their self-driving car simulator. Download data from this [link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

Prepare data for training and testing
```bash
python prepare_data.py path/to/data.zip
```

Alternatively you can manually gather data by driving a car using the aforementioned simulator.

## Training

Model is trained on a CPU by default. To train it using GPU follow this [link](https://www.tensorflow.org/install/).

To train the model run
```bash
python model.py
```

Optional arguments are
* -b   --batch-size (defaults to 32)
* -e   --epochs (defaults to 10)
* -s   --steps-per-epoch (defaults to 10000)
* -l   --learning-rate (defaults to 0.0001)

## Testing

To see how well model performs we use coefficient of determination
```bash
python validation.py path/to/model
```

If you would like to see the model in action you first need to start Udacity simulator in autonomous mode.
Then run
```bash
python drive.py path/to/model
```

## References

* Nvidia paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
* Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
* Implementation details https://github.com/naokishibuya/car-behavioral-cloning
