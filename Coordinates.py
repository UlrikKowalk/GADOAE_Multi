import numpy as np
import torch


class Coordinates:

    def __init__(self, device, num_sensors, dimensions, max_uncertainty=0.0):
        self.device = device
        self.num_sensors = num_sensors
        self.dimensions = dimensions
        self.uncertainty = torch.zeros(size=(self.num_sensors, 3), device=self.device)

        uncertainty_direction = torch.rand(self.num_sensors, device=self.device) * 2 * torch.pi

        for sensor in range(self.num_sensors):
            new_coordinates = [max_uncertainty * torch.cos(uncertainty_direction[sensor]),
                               max_uncertainty * torch.sin(uncertainty_direction[sensor]),
                               0]
            self.uncertainty[sensor, :self.dimensions] = torch.tensor(new_coordinates[:self.dimensions])

    def generate(self, coordinates):
        return torch.add(coordinates, self.uncertainty)
