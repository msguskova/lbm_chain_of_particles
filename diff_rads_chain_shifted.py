import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from numba import njit, prange, set_num_threads

plt.rcParams.update({'font.size':14, 'savefig.facecolor':'w'})

parser = argparse.ArgumentParser(
                    prog='diff_rads_chain_shifted',
                    description='''Chain of rigid particles flows in Poiseuille. \
                    Particles of the equally distributed over length of the channel.\
                    Particles of two radii are alternating in the chain\
                    Interparticle distance can be set, the default value is 4R.''',
                    epilog='Do your duty, come what may.')
                    
parser.add_argument('width', metavar='W', type=int,
                    help='width of the channel in number of nodes (41 will give w = 40)')
                    
parser.add_argument('cs2', metavar='cs2', type=int, choices=[1, 2],
                    help='sound speed of the model')
                
parser.add_argument('u_flow', metavar='u_flow', type=float,
                    help='max velocity for poiseuille flow')
                    
parser.add_argument('nu', metavar='nu', type=float,
                    help='kinematic viscosity')

parser.add_argument('number_of_particles', metavar='N', type=int,
                    help='number of particles in the chain')

parser.add_argument('cy0', metavar='cy0', type=float,
                    help='y-coordinate for initial starting position')
                    
parser.add_argument('rad2', metavar='rad2', type=float,
                    help='Radius of bigger particle')

parser.add_argument('-i', '--interparticle_distance',
                    type=float,
                    help='Distance between particles centers measured in radius of the particles',                  
                    nargs='?', default=4.)
                    


@njit(parallel=False)
def macroscopic(fin, rho0, v, nx, ny):
    rho = np.zeros((nx, ny))#np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in prange(9):
        for ix in range(nx):
            for iy in range(ny):
                u[0, ix, iy] += v[i, 0] * fin[i, ix, iy]
                u[1, ix, iy] += v[i, 1] * fin[i, ix, iy]
                rho[ix, iy] += fin[i, ix, iy]
    u /= rho
    return rho, u

@njit(parallel=False)
def inc_macroscopic(fin, rho0, v, nx, ny):
    rho = np.zeros((nx, ny))#np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in prange(9):
        for ix in range(nx):
            for iy in range(ny):
                u[0, ix, iy] += v[i, 0] * fin[i, ix, iy]
                u[1, ix, iy] += v[i, 1] * fin[i, ix, iy]
                rho[ix, iy] += fin[i, ix, iy]
    u /= rho0
    return rho, u

@njit(parallel=False)
def equilibrium(rho, rho0, u, t, v, cs2, nx, ny):
    feq = np.zeros((9, nx, ny))
    u2 = (u[0] ** 2 + u[1] ** 2) / 2 / cs2
    
    for i in prange(9):
        for ix in range(nx):
            for iy in range(ny):
                u_ci = u[0, ix, iy] * v[i, 0] + u[1, ix, iy] * v[i, 1] 
                feq[i, ix, iy] = t[i] * rho[ix, iy] * (1 +
                                                       u_ci / cs2 - 
                                                       u2[ix, iy] + 
                                                       u_ci ** 2 / 2 / cs2 ** 2)
    return feq

@njit(parallel=False)
def inc_equilibrium(rho, rho0, u, t, v, cs2, nx, ny):
    feq = np.zeros((9, nx, ny))
    u2 = (u[0] ** 2 + u[1] ** 2) / 2 / cs2
    
    for i in prange(9):
        for ix in range(nx):
            for iy in range(ny):
                u_ci = u[0, ix, iy] * v[i, 0] + u[1, ix, iy] * v[i, 1] 
                feq[i, ix, iy] = t[i]  * (rho[ix, iy] +  
                                          rho0 * (u_ci / cs2 - 
                                                  u2[ix, iy] + 
                                                  u_ci ** 2 / 2 / cs2 ** 2))
    return feq

def poiseuille(d, x, y, uw, ny):
    return (1 - d) * 4 * uw * (1 - y / (ny-1)) * (y / (ny-1))

@njit()
def streaming_periodic_x_cs1_3(fin, fout):
    fin[8, :-1, :-1] = fout[8, 1:, 1:]
    fin[8, -1, :-1] = fout[8, 0, 1:]
    
    fin[7, :-1, :] = fout[7, 1:, :]
    fin[7, -1, :]= fout[7, 0, :]
    
    fin[6, :-1, 1:] = fout[6, 1:, :-1]
    fin[6, -1, 1:] = fout[6, 0, :-1]
    
    
    fin[5, :, :-1] = fout[5, :, 1:]
    fin[4] = fout[4]
    fin[3, :, 1:] = fout[3, :, :-1]
    
    fin[2, 1:, :-1] = fout[2, :-1, 1:]
    fin[2, 0, :-1] = fout[2, -1, 1:]
    
    fin[1, 1:, :] = fout[1, :-1, :]
    fin[1, 0, :] = fout[1, -1, :]
    
    fin[0, 1:, 1:] = fout[0, :-1, :-1]    
    fin[0, 0, 1:] = fout[0, -1, :-1] 
    
@njit()    
def bounceback1_cs1_3(fin, fout, ny):    
    fin[2, :, ny-1] = fout[6, :, ny-1]
    fin[5, :, ny-1] = fout[3, :, ny-1]
    fin[8, :, ny-1] = fout[0, :, ny-1]
    fin[0, :, 0] = fout[8, :, 0]
    fin[3, :, 0] = fout[5, :, 0]
    fin[6, :, 0] = fout[2, :, 0]

@njit()
def streaming_periodic_x_cs2_3(fin, fout):
    fin[8, :-1, :-1] = fout[8, 1:, 1:]
    fin[8, -1, :-1] = fout[8, 0, 1:]
    fin[7, :-2, :] = fout[7, 2:, :]
    fin[7, -1, :]= fout[7, 1, :]
    fin[7, -2, :]= fout[7, 0, :]
    fin[6, :-1, 1:] = fout[6, 1:, :-1]
    fin[6, -1, 1:] = fout[6, 0, :-1]
    fin[5, :, :-2] = fout[5, :, 2:]
    fin[4] = fout[4]
    fin[3, :, 2:] = fout[3, :, :-2]
    fin[2, 1:, :-1] = fout[2, :-1, 1:]
    fin[2, 0, :-1] = fout[2, -1, 1:]
    fin[1, 2:, :] = fout[1, :-2, :]
    fin[1, 0, :] = fout[1, -2, :]
    fin[1, 1, :] = fout[1, -1, :]
    fin[0, 1:, 1:] = fout[0, :-1, :-1]    
    fin[0, 0, 1:] = fout[0, -1, :-1] 
    
    
@njit()    
def bounceback1_cs2_3(fin, fout, ny):    
    fin[2, :, ny-1] = fout[6, :, ny-1]
    fin[5, :, ny-1] = fout[3, :, ny-2]
    fin[5, :, ny-2] = fout[3, :, ny-1]
    fin[8, :, ny-1] = fout[0, :, ny-1]
    fin[0, :, 0] = fout[8, :, 0]
    fin[3, :, 0] = fout[5, :, 1]
    fin[3, :, 1] = fout[5, :, 0]
    fin[6, :, 0] = fout[2, :, 0]
    
    
@njit(parallel=False)
def apply_poiseuille_force(fin, force, nx, ny, t, v, cs2):
    for i in prange(9):
        for ix in range(nx):
            for iy in range(ny):
                fin[i, ix, iy] += t[i] * (v[i, 0] * force[0] + 
                                          v[i, 1] * force[1]) / cs2

@njit(parallel=False)
def apply_g_force(fin, g, t, v, cs2):
    ind = np.argwhere(np.logical_or(g[0] != 0, g[1] != 0))    
    for i in prange(9):
        for ix, iy in ind:
            fin[i, ix, iy] += t[i] * (v[i, 0] * g[0, ix, iy] + 
                                      v[i, 1] * g[1, ix, iy]) / cs2

@njit()
def calc_w(r_, L):
    r = abs(r_)
    if r > L / 2:
        r_ -= np.sign(r_) * L
    r = abs(r_)
    if r <= 1:
        return 1 / 8 * (3 - 2 * r + 
                     (1 + 4 * r - 4 * r ** 2) ** 0.5)
    if r <= 2 and r > 1:
        return 1 / 8 * (5 - 2 * r - 
                     (-7 + 12 * r - 4 * r ** 2) ** 0.5)

    return 0.

@njit()
def ibm(particle, u_star, rho, delta_v, L, N, ibm_iterations=6):
    cx, cy = particle[-1, 0], particle[-1, 1]
    N = particle.shape[0] - 1
    # step 0
    gl_lp = np.zeros((N, 2))
    for i in range(N):
        x, y = particle[i, :2]
        
        u_star_lp = np.zeros(2, dtype=np.float64)
        for ixx in range(int(x) - 3, int(x) + 3):
            ix = ixx % L
            for iy in range(int(y) - 3, int(y) + 3):
                w = calc_w(ix - x, L) * calc_w(iy - y, L)
                u_star_lp[0] += w * u_star[0, ix, iy]
                u_star_lp[1] += w * u_star[1, ix, iy]
                #u_star_lp += calc_w(ix - x) * calc_w(iy - y) * u_star[:, ix, iy]
        
        gl_lp[i] = particle[i, 2:4] - u_star_lp 
    
    ul = np.zeros_like(u_star)
    
    for m in range(ibm_iterations):
        # step 1
        gl = np.zeros_like(u_star)
        for i in range(N):
            x, y = particle[i, 0], particle[i, 1]
            
            for ixx in range(int(x) - 3, int(x) + 3):
                ix = ixx % L
                for iy in range(int(y) - 3, int(y) + 3):
                    w = calc_w(ix - x, L) * calc_w(iy - y, L) * delta_v
                    gl[0, ix, iy] += gl_lp[i, 0] * w
                    gl[1, ix, iy] += gl_lp[i, 1] * w
                    
        # step 2            
        for i in range(N):
            x, y = particle[i, 0], particle[i, 1]
            
            for ixx in range(int(x) - 3, int(x) + 3):
                ix = ixx % L
                for iy in range(int(y) - 3, int(y) + 3): 
                    ul[0, ix, iy] = u_star[0, ix, iy] + gl[0, ix, iy]
                    ul[1, ix, iy] = u_star[1, ix, iy] + gl[1, ix, iy]
                    #ul[:, ix, iy] = u_star[:, ix, iy] + gl[:, ix, iy]
                    
        # step 3
        ul_lp = np.zeros((N, 2))
        for i in range(N):
            x, y = particle[i, 0], particle[i, 1]
            for ixx in range(int(x) - 3, int(x) + 3):
                ix = ixx % L
                for iy in range(int(y) - 3, int(y) + 3):
                    w = calc_w(ix - x, L) * calc_w(iy - y, L)
                    ul_lp[i, 0] += ul[0, ix, iy] * w
                    ul_lp[i, 1] += ul[1, ix, iy] * w
                    #ul_lp[i] += ul[:, ix, iy] * calc_w(ix - x) * calc_w(iy - y)
        
        # step 4
        for i in range(N):
            gl_lp[i, 0] += particle[i, 2] - ul_lp[i, 0]
            gl_lp[i, 1] += particle[i, 3] - ul_lp[i, 1]
            #gl_lp[i] += particle[i, 2:4] - ul_lp[i]
        
    #print()
    total_force = np.array([np.sum(gl[0]), np.sum(gl[1])])
    
    total_torque = 0
    
    ind = np.argwhere(np.logical_or(gl[0] != 0, gl[1] != 0))

    for i in ind:
        ix, iy = i
        rx = ix - cx
        if abs(rx) > L / 2:
            rx -= np.sign(rx) * L
        
        total_torque += gl[1, ix, iy] * rx - gl[0, ix, iy] * (iy - cy)
                        
    return total_force, total_torque, gl
            
@njit()
def euler(particle, w, pforce, torque, mass, L, rad):
    particle[-1, 2:4] += pforce / mass
    w += 2 * torque / mass / rad ** 2
    # print(w, torque, 2 * torque / mass / rad ** 2)
    for i in range(particle.shape[0] - 1):
        r = particle[i, :2] - particle[-1, :2]
        if np.abs(r[0]) > L / 2:
            r[0] -= np.sign(r[0]) * L
        
        particle[i, 2] = particle[-1, 2] - r[1] * w
        particle[i, 3] = particle[-1, 3] + r[0] * w
        
    particle[:, :2] += particle[:, 2:4]
    particle[:, :1] %= L
    
    return w
    
#@njit()
def add_repulsive_forces(particle, particle2, pforce1, pforce2, rad1, rad2, L, k=1, delta=2):
    c1 = particle[-1, :2]
    c2 = particle2[-1, :2]

    dist = c1 - c2
    if abs(dist[0]) > L / 2:
        sign = np.sign(dist[0])
        #dist[0] -= np.sign(dist[0]) * L / 2
        dist[0] = L - abs(dist[0])
        dist[0] *= - sign

    mod = (dist[0] ** 2 + dist[1] ** 2) ** 0.5
    #print(mod, end=' ')
    h = mod - (rad1 + rad2)
    assert h > 0, 'particles collided h=%f mod=%f c1=%f c2=%f dist=[%f, %f]' % \
                (h, mod, c1[0], c2[0], dist[0], dist[1])
    if h < delta:
        #print('h=', h, 'delta=', delta, 'dist', dist)
        f12 = k * (delta - h) * dist / mod
        # pforce is from ibm method and
        # will be applied with minus
        # so repulsive force added with opposite sign
        pforce1 -= f12
        pforce2 += f12

        
#@njit()
def add_repulsive_forces_walls(particle, pforce1, rad1, ny, k=1, delta=2):     
    c1 = particle[-1, :2]   
    # bottom wall
    h = c1[1] - rad1
    assert h > 0, '%f particle crossed bottom wall' % c1[1]
    if h < delta:
        f = np.array([0, k * (delta - h)])
        pforce1 -= f

    # top wall
    h = ny - 1 - (c1[1] + rad1)
    assert h > 0, '%f particle crossed top wall' % c1[1]
    if h < delta:
        f = np.array([0, - k * (delta - h)])
        pforce1 -= f
        

def make_particle(cx, cy, uw, ny, N, L, rad, vel_poiseuille=True):
    a = np.linspace(0, 2 * np.pi, N, endpoint=False)
    particle = np.zeros((N + 1, 4), dtype=np.float64)
    particle[:-1, 0] = rad * np.cos(a) + cx
    particle[:-1, 1] = rad * np.sin(a) + cy
    particle[-1, 0], particle[-1, 1] = cx, cy 
    if vel_poiseuille:
        particle[:, 2] = poiseuille(0, 0, cy, uw, ny)
        
    particle[:, :1] %= L
    print(poiseuille(0, 0, cy, uw, ny))
    return particle

def get_data(time, particle, w, pforce, torque):
    return [time, particle[-1, 0], particle[-1, 1],
                             particle[0, 0], particle[0, 1],
                             particle[-1, 2], particle[-1, 3],
                             pforce[0], pforce[1],
                             torque,
                             w
                            ]

def save_particles(time, particles, w, pforce, torque, folder):
    for i in range(len(particles)):
        data = get_data(time, particles[i], w[i], pforce[i], torque[i])

        with open(folder+'disk_re_0.25_%d.txt'%i, 'a') as file:
            np.savetxt(file, np.array([data]))

            
def save_pic(time, u, rho, particles, folder, ny, nx):
    fig, ax = plt.subplots(figsize=(16, 3))
    unorm = np.sqrt(u[0]**2+u[1]**2)
    print(time, unorm[:, ny//2].max(),
          unorm[:, ny//2].max() - unorm[:, ny//2].min(), 
         rho.max(), rho.min())
    #print(time//fig_out, end=' ')
    c = ax.imshow(unorm.transpose(), cmap=cm.coolwarm)
    for p in range(len(particles)):
        ax.plot(particles[p][:-1, 0], particles[p][:-1, 1], '.k', markersize=0.3)
        ax.plot(particles[p][0, 0], particles[p][0, 1], '.w')
    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    plt.colorbar(c, ax=ax)
    fig.tight_layout()
    fig.savefig(folder+'vel %07d.png'%(time))
    plt.close()
    
def save_velocity(time, u, particles, folder):
    for i, particle in enumerate(particles):
        data = [time, particle[-1, 0], particle[-1, 1], 
                     (particle[-1, 2] ** 2 + particle[-1, 3] ** 2) ** 0.5]
        unorm = (u[0, :, int(particle[-1, 1])] ** 2 +  u[1, :, int(particle[-1, 1])] ** 2) ** 0.5
        data += list(unorm)

        with open(folder+'disk_re_0.25_%d_rel_velocity.txt'%i, 'a') as file:
            np.savetxt(file, np.array([data]))
    

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    cs2 = args.cs2 / 3
    cs1_3 = cs2 < 0.5
    
    uw = args.u_flow
    nu = args.nu
    
    rho_particle = 1.
    
    tau = nu / cs2 + 1 / 2
    omega = 1 / tau
    
    inc = 'inc'
    inc_flag = True #if inc== 'inc' else False
    
    D = args.width#80
    Ds = 20
    rad = Ds / 2
    rad2 = args.rad2
    
    cy = args.cy0* (D - 1)
    
    Re = uw * rad / nu
    num_particles = args.number_of_particles
    inter_particle = args.interparticle_distance
    
    L = int(inter_particle * int(rad) * num_particles)
    nx, ny = L, D
    cx = L / 2

    maxIter = 8000000
    fig_out = 8000

    force = np.array([8 * uw * nu / (D - 1) ** 2, 0])
    
   
    #rho_particle = 1000. / 970.
    ibm_iter = 5
    
    folder = 'tmp_chain2_cs2_%.2f_re_%.2f_nu_%.2f_u_%.2f_tau_%.2f_cy_%.2f_W_%d_N_%d_rad2_%.1f_inter_%.1fR/'%\
                                                             (cs2, Re, nu, \
                                                              uw, tau, cy, D, num_particles, \
                                                              rad2, inter_particle)
    
    os.makedirs(folder, exist_ok=True)
    
    print(sys.argv[0])
    #print('cp ' + sys.argv[0] + ' ' + folder + sys.argv[0])
    #os.system('cp ' + sys.argv[0] + ' ' + folder + sys.argv[0])
    shutil.copyfile(sys.argv[0], folder + sys.argv[0])
    
    with open(folder + 'log.data', 'w') as file:      
            print('command: python ' + ' '.join(sys.argv), file=file)  
            print('cs2 =', cs2, file=file)
            print('incompressible =', inc, file=file)
            print('nx =', nx, file=file)
            print('ny =', ny, file=file)
            print('tau =', tau, file=file)
            print('omega =', omega, file=file)
            print('u_flow =', uw, file=file)
            print('nu =', nu, file=file)
            print('Re =', Re, file=file)
            print('cx =', cx, file=file)
            print('cy =', cy, file=file)
            print('ibm_iterations =', ibm_iter, file=file)
            print('rad =', rad, file=file)
            #print('ibm points =', N, file=file)
            print('force =', force, file=file)
            print('rho_particle =', rho_particle, file=file)
            
    with open(folder + 'log.data', 'r') as file:   
        lst = file.readlines()
        for line in lst:
            print(line.strip())
            
            
    if cs1_3:
        v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0],\
              [0, -1], [-1, 1], [-1, 0], [-1, -1]])
        t = np.array([1 / 36, 1 / 9, 1 / 36, 1 / 9, 
                      4 / 9, 
                      1 / 9, 1 / 36, 1 / 9, 1 / 36]) 
    else:
        v = np.array([[1, 1], [2, 0], [1, -1], [0, 2], [0, 0],\
              [0, -2], [-1, 1], [-2, 0], [-1, -1]])
        t = np.array([1 / 9, 1 / 36, 1 / 9, 1 / 36,
                      4 / 9,
                      1 / 36, 1 / 9, 1 / 36, 1 / 9])
        
    if cs1_3:
        bounceback1 = bounceback1_cs1_3
        streaming_periodic_x = streaming_periodic_x_cs1_3
    else:
        bounceback1 = bounceback1_cs2_3
        streaming_periodic_x = streaming_periodic_x_cs2_3
        
    if inc_flag:
        get_rho_u = inc_macroscopic
        get_feq = inc_equilibrium
    else:
        get_rho_u = macroscopic
        get_feq = equilibrium
        
    
    #a = np.linspace(0, 2 * np.pi, N, endpoint=False)

    w = np.array([0] * num_particles, dtype=np.float64)
    #mass = np.pi * rad ** 2 * rho_particle

    #length = 2 * np.pi * rad 
    # N = particle.shape[0] - 1
    lst_delta_v = []
    lst_length = []
    lst_mass = []
    lst_N = []
    lst_rad = []
    
    particles = []
    for i in range(num_particles):
        rad_p = rad
        if i % 2 == 0:
            rad_p = rad2
            
        lst_rad.append(rad_p)
        N =  int(1.3 * (2 * rad_p) * np.pi * 1.)
        lst_N.append(N)
        length_particle = 2 * np.pi * rad_p 
        lst_length.append(length_particle)
        lst_delta_v.append(length_particle / N)
        lst_mass.append(np.pi * rad_p ** 2 * rho_particle)
        rx, ry = np.random.rand() - 0.5, np.random.rand() - 0.5
        amp = 2
        particles.append(make_particle(inter_particle * rad * i + amp * rx, cy + amp * ry, 
                                        uw, ny, N, L, rad_p, vel_poiseuille=False))

        
    
    rho = np.ones((nx, ny))
    rho0 = 1.

    u = np.fromfunction(poiseuille, (2, nx, ny), uw=uw, ny=ny)
    fin = get_feq(rho, rho0, u, t, v, cs2, nx, ny)

    lst = []
    lst_particle = []

    pforce = np.zeros((num_particles, 2))
    torque = np.zeros(num_particles)        
    
    for time in range(0, maxIter):
        rho, u = get_rho_u(fin, rho0, v, nx, ny)
        feq = get_feq(rho, rho0, u, t, v, cs2, nx, ny)

        if time > 1:
            bounceback1(fin, fout, ny)   

        #collision step
        fout = fin * (1 - omega) + omega * feq

        streaming_periodic_x(fin, fout)

        rho, u = get_rho_u(fin, rho0, v, nx, ny)
        g = np.zeros_like(u)
        for p in range(num_particles):
            pforce[p], torque[p], g_tmp = ibm(particles[p], u, 
                                              rho, lst_delta_v[p], 
                                              L, lst_N[i], ibm_iterations=ibm_iter)
            g += g_tmp
            add_repulsive_forces_walls(particles[p], 
                                 pforce[p], 
                                 lst_rad[p], ny)
                                 
            for pp in range(p - 1, p):
                add_repulsive_forces(particles[p], particles[pp],
                                 pforce[p], pforce[pp], 
                                 lst_rad[p], lst_rad[pp], L)
    
        for p in range(num_particles):
            w[p] = euler(particles[p], w[p], -pforce[p], -torque[p], 
                         lst_mass[p], L, lst_rad[p])

        #force
        apply_poiseuille_force(fin, force, nx, ny, t, v, cs2)
        apply_g_force(fin, g, t, v, cs2)
        #break

        if np.any(np.isnan(u)):
            print(time, 'nan')
            break

        if(time % 10 == 0):
            save_particles(time, particles, w, pforce, torque, folder)

        if (time % fig_out == 0):
        #if (time % 1 == 0):
            np.save(folder+'checkpoint_fin.npy', fin)
            save_pic(time, u, rho, particles, folder, ny, nx)
            #save_velocity(time, u, particles, folder)
        
    
    
