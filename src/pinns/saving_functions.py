import pickle
import numpy as np

################################################################################
################################## Saving functions ############################
################################################################################


def save_files(
    loss_head,
    head,
    m: int,
    initial_x: float,
    final_t: float,
    alpha_: float,
    width_base: int,
    number_of_epochs: int,
    grid_size: int,
    tl: str = "",
) -> None:
    """Save loss and trajectory data to pickle files.
    
    Args:
        loss_head: 
            Loss values for this head
        head: 
            Neural network head outputs
        m: 
            Head index
        initial_x: 
            Initial x position
        final_t: 
            Final time
        alpha_: 
            Potential strength
        width_base: 
            Base network width
        number_of_epochs: 
            Number of training epochs
        grid_size: 
            Grid size for training points
        tl: 
            Transfer learning suffix (default: "")
    """
    base_filename = (
        f"Data/Head_{m}_Initial_x_{initial_x}_final_t_{final_t}_"
        f"alpha_{alpha_}_width_base_{width_base}_number_of_epochs{number_of_epochs}_"
        f"grid_size_{grid_size}"
    )
    # Save loss
    with open(f"{base_filename}loss_individual{tl}.p", "wb") as f:
        pickle.dump(loss_head, f)
    # Save trajectories
    components = {
        'x': 0,
        'y': 1,
        'px': 2,
        'py': 3
    }
    for component, idx in components.items():
        with open(f"{base_filename}_Trajectory_NN_{component}{tl}.p", "wb") as f:
            pickle.dump(head.cpu().detach()[:, idx], f)


def save_file_numerical(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray, initial_x: float, final_t: float, alpha_: float, tl: str = ""):
    """Save numerical trajectories to pickle files.
    
    Args:
        x, y, px, py: 
            Trajectory components
        initial_x: 
            Initial x position
        final_t: 
            Final time
        alpha_: 
            Potential strength
        tl: 
            Transfer learning suffix (default: "")
    """
    # Dictionary mapping variable names to their values
    trajectories = {
        'x': x,
        'y': y,
        'px': px,
        'py': py
    }
    base_filename = f"Data/Initial_x_{initial_x}_final_t_{final_t}_alpha_{alpha_}"
    for var_name, data in trajectories.items():
        filename = f"{base_filename}_numerical_trajectories_{var_name}{tl}.p"
        with open(filename, "wb") as f:
            pickle.dump(data, f)