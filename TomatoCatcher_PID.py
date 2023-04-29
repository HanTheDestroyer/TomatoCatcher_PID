import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


# time variables
dt = 0.1
time = np.arange(0, 60 + dt, dt)
frame_count = len(time)

# basic simulation variables
rail_angle = np.pi / 48
cart_mass = 10
gravity = 0.5

# controller variables
error = np.zeros(frame_count)
error_derivative = np.zeros(frame_count)
error_integral = np.zeros(frame_count)
kp, kd, ki = 12, 24, 1

# controlled variables
cart_position_x = np.ones(frame_count) * np.random.randint(100, 900)
cart_position_y = cart_position_x * np.sin(rail_angle)
tomato_position_y = np.ones(frame_count, dtype=float) * np.random.randint(100, 180)
tomato_position_x = np.random.randint(100, 900) * np.ones(frame_count)
cart_velocity = np.zeros(frame_count, dtype=float)
cart_velocity_x = np.zeros(frame_count, dtype=float)
cart_velocity_y = np.zeros(frame_count, dtype=float)
cart_acceleration = np.zeros(frame_count, dtype=float)

# CONTROLLER LOOP
for i in range(frame_count):
    if i == 0:
        error[i] = tomato_position_x[i] - cart_position_x[i]
    else:
        # various error values
        error[i] = tomato_position_x[i] - cart_position_x[i-1]
        error_derivative[i] = (error[i] - error[i - 1]) / dt
        error_integral[i] = error_integral[i - 1] + error[i] * dt
        # system physics
        controller_force = kp * error[i] + kd * error_derivative[i] + ki * error_integral[i]
        net_force = controller_force + gravity * cart_mass * np.sin(rail_angle)
        cart_acceleration[i] = net_force / cart_mass
        cart_velocity[i] = cart_velocity[i - 1] + 0.5 * dt * (cart_acceleration[i - 1] + cart_acceleration[i])
        cart_velocity_x[i] = cart_velocity[i] * np.cos(rail_angle)
        cart_velocity_y[i] = cart_velocity[i] * np.sin(rail_angle)
        cart_position_x[i] = cart_position_x[i - 1] + 0.5 * dt * (cart_velocity_x[i - 1] + cart_velocity_x[i])
        cart_position_y[i] = cart_position_y[i - 1] + 0.5 * dt * (cart_velocity_y[i - 1] + cart_velocity_y[i])

        # if distance is smaller than a set value, tomato is caught
        tomato_position = np.array([tomato_position_x[i-1], tomato_position_y[i-1]], dtype=float)
        cart_position = np.array([cart_position_x[i-1], cart_position_y[i-1]], dtype=float)
        distance = np.linalg.norm(tomato_position - cart_position)
        distance_x = np.abs(tomato_position[0] - cart_position[0])
        distance_y = np.abs(tomato_position[1] - cart_position[1])
        if distance_x > 25 or distance_y > 10:
            tomato_position_y[i] -= 0.5 * gravity * time[i] ** 2
        else:
            tomato_position_y[i] = tomato_position_y[i-1]


def update_plot(t):
    cart.set_data([cart_position_x[t], cart_position_x[t]+20], [cart_position_y[t], cart_position_y[t]])
    tomato.set_data([tomato_position_x[t], tomato_position_x[t]+10], [tomato_position_y[t], tomato_position_y[t]])
    pos_tracking_x.set_data(time[0:t], cart_position_x[0:t])
    error_tracking.set_data(time[0:t], error[0:t])
    vel_tracking.set_data(time[0:t], cart_velocity_x[0:t])

    return cart, tomato, pos_tracking_x, error_tracking, vel_tracking


# create plots, start with figure
np.set_printoptions(suppress=True)
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=[0.4, 0.4, 0.4])
gs = gridspec.GridSpec(4, 3)

# figure 0
ax0 = fig.add_subplot(gs[0:2, :], facecolor=[0.6, 0.6, 0.6])
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
plt.xlim(0, 1000)
plt.ylim(0, 200)
rail_x = np.arange(0, 1001, 1)
rail_y = np.arange(0, 1001, 1) * np.sin(rail_angle)
rail = ax0.plot(rail_x, rail_y, 'k', linewidth=2)
cart = ax0.plot([], [], 'k', linewidth=10)[0]
tomato = ax0.plot([], [], 'r', linewidth=20)[0]

# figure 1
ax1 = fig.add_subplot(gs[2:4, 0], facecolor=[0.6, 0.6, 0.6])
plt.xlabel('Time', color=(0, 1, 0))
plt.ylabel('X-Position', color=(0, 1, 0))
plt.xlim(0, time[-1])
plt.ylim(-200, 1200)
pos_desired = ax1.plot(time, tomato_position_x * np.ones(frame_count), 'r', linewidth=1)
pos_tracking_x = ax1.plot([], [], linewidth=3)[0]

# figure 2
ax2 = fig.add_subplot(gs[3, 1], facecolor=[0.6, 0.6, 0.6])
plt.xlabel('Time', color=(1, 0, 0))
plt.ylabel('Error', color=(1, 0, 0))
plt.xlim(0, time[-1])
plt.ylim(-200, 200)
error_tracking = ax2.plot([], [])[0]

# figure 3
ax3 = fig.add_subplot(gs[2:4, 2], facecolor=[0.6, 0.6, 0.6])
plt.xlabel('Time', color=(0, 0, 1))
plt.ylabel('Velocity', color=(0, 0, 1))
plt.xlim(0, time[-1])
plt.ylim(-100, 100)
vel_tracking = ax3.plot([], [])[0]

volume_animation = animation.FuncAnimation(fig, update_plot, frames=frame_count,
                                           interval=2, repeat=True, blit=True)
plt.show()
