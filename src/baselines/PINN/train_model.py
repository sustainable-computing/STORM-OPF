import numpy as np
import argparse
from src.baselines.PINN.create_example_parameters import create_example_parameters
from src.baselines.PINN.create_data import create_data
from src.baselines.PINN.PinnModel import PinnModel
from src.baselines.PINN import generate_data
from src import util
import pickle

def train_model(net, gen_data, case):
    
    simulation_parameters = create_example_parameters(net)
    num_bus= args.case[4:]
    if gen_data:
        #generate new train data and create the necessary .csv files
        generate_data.generate_training_data(num_bus, case)
    
    #load pre-saved train data ( .csv files)
    x_training, y_training, Lg_Max = create_data(net, simulation_parameters=simulation_parameters)

    simulation_parameters.update({'Lg_Max':Lg_Max})
    # Weights of Lg and PINN
    lw1=0.05
    lw2=0.05
    model = PinnModel(lw1,lw2,simulation_parameters=simulation_parameters)
    np.set_printoptions(precision=3)
    print('Starting training')

    for n_epochs, batch_size in zip(simulation_parameters['training']['epoch_schedule'],
                                simulation_parameters['training']['batching_schedule']):

        model.fit(x_training,
                  y_training,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True)
    
    model.save("src/baselines/PINN/trained_model/" + num_bus + "/")
    #save simulation param because we need it for test later
    sim_param_path= "src/baselines/PINN/datasets/" + num_bus + "/simulation_parameters.pkl"
    with open(sim_param_path, "wb") as f:
        pickle.dump(simulation_parameters, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case14")
    parser.add_argument("--gen_data", action='store_true', default=False, help='Want to generate new train data?')
    args = parser.parse_args()

    base_net, _ = util.get_base_net(args.case, 'in_distribution')
    train_model(base_net, args.gen_data, args.case)