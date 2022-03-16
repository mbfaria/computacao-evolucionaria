import numpy as np
from typing import Callable
from objective_functions import peaks, rastrigin


class PSOlver:
    def __init__(self,
                 loss_function: Callable,
                 lower_bound: int,
                 upper_bound: int,) -> None:

        self.loss_function = loss_function
        self.LOWER_BOUND = lower_bound
        self.UPPER_BOUND = upper_bound
        self.random_generator = np.random.default_rng()

    def _to_debug(self, variable: dict) -> None:
        variable_name = [name for name in vars(self) if vars(self)[
            name] is variable][0]
        print(70*"_")
        print(f"self.{variable_name}")
        print(70*"_")
        print(variable)
        print(70*"_")

    def _set_constants(self, parameters: dict) -> None:

        self.MAX_ITERATIONS = parameters.get("max_iterations")
        self.SWARM_SIZE = parameters.get("swarm_size")
        self.NUMBER_VARIABLES = parameters.get("number_variables")
        self.MAX_VELOCITY = parameters.get("max_velocity")
        self.INITIAL_INERTIA_WEIGHT = parameters.get(
            "initial_inertia_weight")
        self.ACCELERATION_COGNITIVE = parameters.get(
            "acceleration_cognitive")
        self.ACCELERATION_SOCIAL = parameters.get("acceleration_social")

    def _to_initilize(self) -> None:
        self.positions = self.random_generator.uniform(
            low=self.LOWER_BOUND, high=self.UPPER_BOUND, size=(
                self.SWARM_SIZE, self.NUMBER_VARIABLES)
        )

        self.velocities = self.random_generator.uniform(
            low=-self.MAX_VELOCITY, high=self.MAX_VELOCITY, size=(
                self.SWARM_SIZE, self.NUMBER_VARIABLES)
        )

    def _to_evaluete(self) -> None:
        self.loss = np.array(list(map(self.loss_function, self.positions)))

    def _get_best_position(self, first_call: bool = False) -> None:
        arg_best_loss = np.argmin(self.loss)
        if first_call:
            self.best_global_loss = self.loss[arg_best_loss]
            self.best_global_position = self.positions[arg_best_loss]
            self.best_individual_loss = self.loss
            self.best_individual_position = self.positions
        else:
            if self.loss[arg_best_loss] < self.best_global_loss:
                self.best_global_loss = self.loss[arg_best_loss]
                self.best_global_position = self.positions[arg_best_loss]

            arg_best_loss_individual = self.loss < self.best_individual_loss
            self.best_individual_loss[arg_best_loss_individual] = (
                self.loss[arg_best_loss_individual]
            )
            self.best_individual_position[arg_best_loss_individual] = (
                self.positions[arg_best_loss_individual]
            )

    def _to_update_velocities(self) -> None:
        random_variables = self.random_generator.uniform(size=2)

        self.velocities = (self.inertia_weight * self.velocities) + \
            (random_variables[0]*self.ACCELERATION_COGNITIVE *
             (self.best_individual_position-self.positions)) + \
            (random_variables[1]*self.ACCELERATION_SOCIAL *
             (self.best_global_position-self.positions))

        trespassing_max_speed = np.abs(self.velocities) > self.MAX_VELOCITY

        while trespassing_max_speed.any():
            self.velocities[trespassing_max_speed] = (
                np.sign(self.velocities[trespassing_max_speed])*(
                    np.abs(self.velocities[trespassing_max_speed]) -
                    self.MAX_VELOCITY)
            )
            trespassing_max_speed = np.abs(
                self.velocities) > self.MAX_VELOCITY

    def _to_update_positions(self) -> None:
        self.positions = self.positions + self.velocities

    def _to_update_inertia_weight(self, iteration) -> None:
        self.inertia_weight = self.INITIAL_INERTIA_WEIGHT - \
            (((self.INITIAL_INERTIA_WEIGHT)/self.MAX_ITERATIONS)*iteration)

    def run(self,
            max_iterations: int,
            swarm_size: int,
            number_variables: int,
            max_velocity: float,
            initial_inertia_weight: float,
            acceleration_cognitive: float,
            acceleration_social: float,
            ) -> None:

        self._set_constants(locals())
        self._to_initilize()
        self._to_evaluete()
        self._get_best_position(first_call=True)

        for iteration in range(self.MAX_ITERATIONS):
            print(
                f"[ITERATION {str(iteration).zfill(2)}] "
                f"f(x*)={self.best_global_loss}, "
                f"x*={self.best_global_position}")

            self._to_update_inertia_weight(iteration)
            self._to_update_velocities()
            self._to_update_positions()
            self._to_evaluete()
            self._get_best_position()


if __name__ == "__main__":
    # solver_peaks = PSOlver(peaks, lower_bound=-3, upper_bound=3)
    solver_rastrigin = PSOlver(rastrigin, -2, 2)

    solver_rastrigin.run(max_iterations=20,
                         swarm_size=50,
                         number_variables=2,
                         max_velocity=1.5,
                         initial_inertia_weight=1.5,
                         acceleration_cognitive=1,
                         acceleration_social=0.5)
