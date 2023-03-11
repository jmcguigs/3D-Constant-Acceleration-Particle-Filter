# -*- coding: utf-8 -*-
"""

@author: jemcg

classes for tracking a target undergoing constant acceleration motion in 3D
space using a particle filter
"""

import numpy as np
from matplotlib import pyplot as plt


# A static observer that uses a Sensor to take measurements of a Target
class Observer:
    def __init__(self) -> None:
        self.position: np.array = np.array([0, 0, 0])

    def setPosition(self, x: float, y: float, z: float) -> None:
        self.position = np.array([x, y, z])


# sensor used to detect/measure targets- generates range/bearing estimates
class Sensor:
    def __init__(self, parent: Observer) -> None:
        self.bearingUncertainty: float = 0
        self.elevationUncertainty: float = 0
        self.rangingUncertainty: float = 0
        self.position: np.array = parent.position

    # generate a noisy estimate of a target's range, bearing, & elevation angle
    def measure(self, target) -> list:
        # difference between target and sensor positions
        delta: np.array = target.state[[0, 3, 6]] - self.position

        # length of delta vector gives true range
        rangeEstimate: float = np.sqrt(sum(delta**2))

        # angles to target (bearing from x-axis and elevation from x-y plane)
        bearingEstimate: float = np.arctan2(delta[1], delta[0])
        elevationEstimate: float = np.arcsin(delta[2] / rangeEstimate)

        # add noise (assumed normally distributed and unbiased)
        # range noise based on fractional ranging 1-sigma uncertainty
        rangeEstimate += np.random.normal(0, rangeEstimate *
                                          self.rangingUncertainty)

        # bearing/elevation noise based on angular 1-sigma uncertainties
        bearingEstimate += np.random.normal(0, self.bearingUncertainty)
        elevationEstimate += np.random.normal(0, self.elevationUncertainty)

        return([rangeEstimate, bearingEstimate, elevationEstimate])

    # given bearing uncertainty sigma in degrees, convert to rad and set
    def setBearingUncertainty(self, sigma: float) -> None:
        self.bearingUncertainty = np.pi * sigma / 180

    # given elevation uncertainty sigma in degrees, convert to rad and set
    def setElevationUncertainty(self, sigma: float) -> None:
        self.elevationUncertainty = np.pi * sigma / 180

    # set fractional range uncertainty
    def setRangingUncertainty(self, sigma: float) -> None:
        self.rangingUncertainty = sigma


# object with constant acceleration motion in 3D space
class Target:
    def __init__(self) -> None:
        self.state: np.array = np.transpose(np.zeros(9))

    # set target's x, y, z position (indices 0, 3, 6 in state vector)
    def setPosition(self, x: float, y: float, z: float) -> None:
        self.state[[0, 3, 6]] = [x, y, z]

    # set target's x, y, z velocity (indices 1, 4, 7 in state vector)
    def setVelocity(self, vx: float, vy: float, vz: float) -> None:
        self.state[[1, 4, 7]] = [vx, vy, vz]

    # set target's x, y, z acceleration (indices 2, 5, 8 in state vector)
    def setAcceleration(self, ax: float, ay: float, az: float) -> None:
        self.state[[2, 5, 8]] = [ax, ay, az]

    # propagate the target's state forward by time step dt
    def propagate(self, dt) -> None:
        # 1-dimensional state transition matrix for CA motion
        phi1d: np.array = np.array([[1, dt, 0.5*dt**2],
                                    [0, 1, dt],
                                    [0, 0, 1]])
        o3: np.array = np.zeros([3, 3])

        # 9x3 subarrays to make full 3D state transition matrix
        sub1: np.array = np.hstack((phi1d, o3, o3))
        sub2: np.array = np.hstack((o3, phi1d, o3))
        sub3: np.array = np.hstack((o3, o3, phi1d))

        # 3D state transition matrix for CA motion
        phi3d: np.array = np.vstack((sub1, sub2, sub3))

        # update target's state using past state and state transition matrix
        self.state = np.transpose(np.dot(phi3d, self.state))

    # plot the target's current position on the matplotlib axis ax
    def plot(self, ax) -> None:
        ax.scatter(self.state[0], self.state[3], self.state[6], color='blue')


# a single particle representing a possible target state
class Particle:
    def __init__(self, state: np.array) -> None:
        self.state: np.array = state

    # propagate the particle forward in time with a CA motion model
    def propagate(self, dt: float) -> 'Particle':
        phi1d: np.array = np.array([[1, dt, 0.5*dt**2],
                                    [0, 1, dt],
                                    [0, 0, 1]])

        o3: np.array = np.zeros([3, 3])

        # 9x3 subarrays to make full 3D state transition matrix
        sub1: np.array = np.hstack((phi1d, o3, o3))
        sub2: np.array = np.hstack((o3, phi1d, o3))
        sub3: np.array = np.hstack((o3, o3, phi1d))

        # 3D state transition matrix for CA motion
        phi3d: np.array = np.vstack((sub1, sub2, sub3))

        # return a new particle with updated state vector
        return Particle(np.transpose(np.dot(phi3d, self.state)))


# a particle filter for tracking objects in 3D space with constant acceleration
class ParticleFilter:
    def __init__(self, target: Target, particleCount: int) -> None:
        # target to be tracked
        self.target: Target = target

        # number of particles to generate/propagate
        self.particleCount: int = particleCount

        # array of particles
        self.particles: np.array = np.array([None for i in
                                             range(self.particleCount)])

        # particle weights for resampling
        self.weights: np.array = np.array([1 for i in
                                           range(self.particleCount)])

        # effective particle count (used to avoid particle degeneracy)
        self.effectiveParticleCount: int = self.particleCount

    # generate particles occupying the possible state space given by a set of
    # 3 measurements
    def generateParticles(self, measurements: np.array, sensor: Sensor) -> None:
        # positions in the three measurement distributions
        allPositions: np.array = np.zeros([3, self.particleCount * 3])

        for i in range(3):
            # take the range/bearing/elevation measurement and convert it to
            # 3D cartesian coordinates
            mu: np.array = np.array([np.cos(measurements[1, i]) *
                                     np.cos(measurements[2, i]),
                                     np.sin(measurements[1, i]) *
                                     np.cos(measurements[2, i]),
                                     np.sin(measurements[2, i])]) * measurements[0, i]

            # fill the initial position state space with particles
            centerState: np.array = np.zeros(9)
            centerState[[0, 3, 6]] = mu     # measured position

            # add a particle at the center of the measurement distribution
            distributionCenter: Particle = Particle(centerState)

            # use the 'center' particle to generate the rest of the particles
            offset: int = i * self.particleCount - 1
            for i in range(self.particleCount):
                measi: list[float] = sensor.measure(distributionCenter)
                allPositions[:, i + offset] = np.array([np.cos(measi[1]) *
                                                        np.cos(measi[2]),
                                                        np.sin(measi[1]) *
                                                        np.cos(measi[2]),
                                                        np.sin(measi[2])]) * measi[0]

        # generate particles using estimated pos/vel/acc
        for i in range(self.particleCount):
            index0: int = i                        # index of 1st pos estimate
            index1: int = i + self.particleCount - 1  # 2nd pos estimate
            index2: int = i + (2 * self.particleCount) - 1  # 3rd pos estimate

            # velocity estimates
            v1: np.array = allPositions[:, index1] - allPositions[:, index0]
            v2: np.array = allPositions[:, index2] - allPositions[:, index1]
            a: np.array = v2 - v1                   # acceleration estimate
            pos: np.array = allPositions[:, index2]    # latest position

            ithState: np.array = np.zeros(9)  # empty state vector
            ithState[[0, 3, 6]] = pos   # set position
            ithState[[1, 4, 7]] = v2    # set velocity
            ithState[[2, 5, 8]] = a     # set acceleration

            self.particles[i] = Particle(np.transpose(ithState))

    # propagate particles forward in time
    def propagateParticles(self, dt: float) -> None:
        for i in range(self.particleCount):
            self.particles[i] = self.particles[i].propagate(dt)

    # return mean vector and covariance matrix of a position measurement
    def getDistribution(self, measurement: list, sensor: Sensor) -> list:
        # create a pseudo-target at the last measurement, then use it to
        # generate a distribution of particles around the measurement

        # take the range/bearing/elevation measurement and convert it to
        # 3D cartesian coordinates
        mu: np.array = np.array([np.cos(measurement[1]) *
                                 np.cos(measurement[2]),
                                 np.sin(measurement[1]) *
                                 np.cos(measurement[2]),
                                 np.sin(measurement[2])]) * measurement[0]

        centerState: np.array = np.zeros(9)
        centerState[[0, 3, 6]] = mu     # measured position

        # add a particle at the center of the measurement distribution
        distributionCenter: Particle = Particle(centerState)
        positions = np.zeros([3, self.particleCount])

        # use the 'center' particle to generate the rest of the particles
        for i in range(self.particleCount):
            measi: list[float] = sensor.measure(distributionCenter)
            positions[:, i] = np.array([np.cos(measi[1]) * np.cos(measi[2]),
                                        np.sin(measi[1]) * np.cos(measi[2]),
                                        np.sin(measi[2])]) * measi[0]

        # get covariance matrix of particle positions
        covariance = np.cov(positions)

        # return mean vector and covariance matrix of the measurement
        return mu, covariance

    # calculate particle weights and resample
    def resampleParticles(self, measurement, sensor, keepFraction: float) -> None:
        mu, covariance = self.getDistribution(measurement, sensor)

        particlesToKeep: int = int(keepFraction * self.particleCount)
        particlesToGenerate: int = self.particleCount - particlesToKeep

        for i, p in enumerate(self.particles):
            # calculate mahalanobis distance for each particle
            # (distance of particle from the last measurement distribution)
            dm: float = np.sqrt(np.transpose((p.state[[0, 3, 6]] - mu)) @
                                np.linalg.inv(covariance) @
                                (p.state[[0, 3, 6]] - mu))

            # set weight based on mahalanobis distance (lower weight if
            # particle is further from distribution)
            self.weights[i] = self.weights[i] + self.weights[i] / (1 + dm)

        # normalize to 1.0 total probability
        self.weights = self.weights / sum(self.weights)

        # resample particles based on their weights
        self.particles[0: particlesToKeep] = np.random.choice(self.particles,
                                                              particlesToKeep,
                                                              p=self.weights)

        for i in range(particlesToGenerate):
            # index of a random kept particle
            randomParticleIndex: int = np.random.randint(low=0,
                                                         high=particlesToKeep)

            # grab particle at random index, modify its' state, generate
            # new particle using that state

            # 'nudge' the particle's state in the direction of the
            # measurement
            delta = mu - self.particles[randomParticleIndex].state[[0, 3, 6]]
            modifiedState = self.particles[randomParticleIndex].state
            modifiedState[0] += abs(np.random.normal(0, 0.2)) * delta[0]
            modifiedState[3] += abs(np.random.normal(0, 0.2)) * delta[1]
            modifiedState[6] += abs(np.random.normal(0, 0.2)) * delta[2]

            modifiedState[1] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[1]
            modifiedState[4] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[4]
            modifiedState[7] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[7]

            modifiedState[2] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[2]
            modifiedState[5] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[5]
            modifiedState[8] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[8]

            modifiedState = np.transpose(modifiedState)

            self.particles[self.particleCount - (i + 1)] = Particle(modifiedState)

    # calculate particle weights and resample based on a set of 3 measurements
    def resampleParticles2(self, measurements, sensor, keepFraction: float) -> None:
        allPositions: np.array = np.zeros([3, self.particleCount * 3])

        for i in range(3):
            # take the range/bearing/elevation measurement and convert it to
            # 3D cartesian coordinates
            mu: np.array = np.array([np.cos(measurements[1, i]) *
                                     np.cos(measurements[2, i]),
                                     np.sin(measurements[1, i]) *
                                     np.cos(measurements[2, i]),
                                     np.sin(measurements[2, i])]) * measurements[0, i]

            # fill the initial position state space with particles
            centerState: np.array = np.zeros(9)
            centerState[[0, 3, 6]] = mu     # measured position

            # add a particle at the center of the measurement distribution
            distributionCenter: Particle = Particle(centerState)

            # use the 'center' particle to generate the rest of the particles
            offset: int = i * self.particleCount - 1
            for i in range(self.particleCount):
                measi: list[float] = sensor.measure(distributionCenter)
                allPositions[:, i + offset] = np.array([np.cos(measi[1]) *
                                                        np.cos(measi[2]),
                                                        np.sin(measi[1]) *
                                                        np.cos(measi[2]),
                                                        np.sin(measi[2])]) * measi[0]

        allStates = np.zeros([9, self.particleCount])
        avgState = np.zeros(9)
        # generate particles using estimated pos/vel/acc
        for i in range(self.particleCount):
            index0: int = i                        # index of 1st pos estimate
            index1: int = i + self.particleCount - 1  # 2nd pos estimate
            index2: int = i + (2 * self.particleCount) - 1  # 3rd pos estimate

            # velocity estimates
            v1: np.array = allPositions[:, index1] - allPositions[:, index0]
            v2: np.array = allPositions[:, index2] - allPositions[:, index1]
            a: np.array = v2 - v1                   # acceleration estimate
            pos: np.array = allPositions[:, index2]    # latest position

            allStates[[0, 3, 6], i] = pos
            allStates[[1, 4, 7], i] = v2
            allStates[[2, 5, 8], i] = a

            avgState[[0, 3, 6]] += pos
            avgState[[1, 4, 7]] += v2
            avgState[[2, 5, 8]] += a

        avgState /= self.particleCount
        stateCovariance = np.cov(allStates)

        particlesToKeep: int = int(keepFraction * self.particleCount)
        particlesToGenerate: int = self.particleCount - particlesToKeep

        for i, p in enumerate(self.particles):
            # calculate mahalanobis distance for each particle
            # (distance of particle from the last measurement distribution)
            dm = np.sqrt((p.state[0] - avgState[0])**2 +
                         (p.state[1] - avgState[1])**2 +
                         (p.state[2] - avgState[2])**2)

            # set weight based on mahalanobis distance (lower weight if
            # particle is further from distribution)
            self.weights[i] = self.weights[i] / (1 + dm)

        # normalize to 1.0 total probability
        self.weights = self.weights / sum(self.weights)

        # resample particles based on their weights
        self.particles[0: particlesToKeep] = np.random.choice(self.particles,
                                                              particlesToKeep,
                                                              p=self.weights)

        for i in range(particlesToGenerate):
            # index of a random kept particle
            randomParticleIndex: int = np.random.randint(low=0,
                                                         high=particlesToKeep)

            # grab particle at random index, modify its' state, generate
            # new particle using that state

            # 'nudge' the particle's state in the direction of the
            # measurement
            delta = mu - self.particles[randomParticleIndex].state[[0, 3, 6]]
            modifiedState = self.particles[randomParticleIndex].state
            modifiedState[0] += abs(np.random.normal(0, 0.01)) * delta[0]
            modifiedState[3] += abs(np.random.normal(0, 0.01)) * delta[1]
            modifiedState[6] += abs(np.random.normal(0, 0.01)) * delta[2]

            modifiedState[1] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[1]
            modifiedState[4] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[4]
            modifiedState[7] += np.random.normal(0, 0.01) * self.particles[
                randomParticleIndex].state[7]

            modifiedState[2] += np.random.normal(0, 0.001) * self.particles[
                randomParticleIndex].state[2]
            modifiedState[5] += np.random.normal(0, 0.001) * self.particles[
                randomParticleIndex].state[5]
            modifiedState[8] += np.random.normal(0, 0.001) * self.particles[
                randomParticleIndex].state[8]

            modifiedState = self.target.state
            modifiedState = np.transpose(modifiedState)

            self.particles[self.particleCount - (i + 1)] = Particle(modifiedState)

    # calculate the mean square error of the current position estimate
    def calculateMSE(self, target) -> list:
        positions = np.zeros([3, self.particleCount])
        for i, p in enumerate(self.particles):
            positions[:, i] = p.state[[0, 3, 6]]

        mu = np.array([np.mean(positions[0, :]), np.mean(positions[1, :]),
                       np.mean(positions[2, :])])

        variance = np.array([np.var(positions[0, :]), np.var(positions[1, :]),
                             np.var(positions[2, :])])

        # bias vector of average position estimate
        bias = mu - target.state[[0, 3, 6]]

        # calculate the mean-square error of the position estimate in 3 axes
        MSEx: float = variance[0] + bias[0]**2
        MSEy: float = variance[1] + bias[1]**2
        MSEz: float = variance[2] + bias[2]**2

        return MSEx, MSEy, MSEz

    # plot current particle positions on a matplotlib axis3d (ax)
    def plotParticles(self, ax: plt.Axes) -> None:
        colors = [[0.5, w, 0.2] for w in self.weights]
        positions: np.array = np.zeros([3, self.particleCount])
        for i, p in enumerate(self.particles):
            positions[:, i] = p.state[[0, 3, 6]]

        ax.scatter(positions[0, :], positions[1, :], positions[2, :],
                   color=colors, alpha=0.2)



