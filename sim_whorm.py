import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit, Aer, execute
from scipy.constants import hbar, G, c

class QuantumWormholeModel:
    def __init__(self, majorana_mass=2.5e-25, dark_energy=-1.2e-10, 
                 wormhole_distance=1.6e-35, teleport_time=5.4e-44):
        """
        Inicializar el modelo de teletransportación a través de agujeros de gusano cuánticos
        
        Parámetros:
        - majorana_mass: Masa del fermión de Majorana (kg) - escala de neutrino
        - dark_energy: Energía negativa de la materia oscura (J) - basada en densidad de energía oscura
        - wormhole_distance: Distancia del agujero de gusano (m) - escala de longitud de Planck
        - teleport_time: Tiempo de teletransportación (s) - escala de tiempo de Planck
        """
        self.m = majorana_mass
        self.E = dark_energy
        self.d = wormhole_distance
        self.t_teleport = teleport_time
        
        # Matrices de Dirac (versión simplificada 2D)
        self.alpha_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.alpha_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.beta = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Estado inicial (simplificado)
        self.psi_initial = np.array([1, 0], dtype=complex)
        
    def dirac_equation_rhs(self, t, psi, p_x, p_y):
        """Lado derecho de la ecuación de Dirac para evolución temporal"""
        # Simplificación 2D de la ecuación de Dirac
        H_dirac = p_x * self.alpha_x + p_y * self.alpha_y + self.m * self.beta
        return (-1j / hbar) * np.dot(H_dirac, psi)
    
    def schrodinger_evolution(self, t_span, num_points=1000):
        """Evolución temporal según la ecuación de Schrödinger"""
        # Parámetros simplificados para este modelo conceptual
        p_x = hbar / self.d
        p_y = 0
        
        # Resolver la ecuación diferencial
        sol = solve_ivp(
            lambda t, y: self.dirac_equation_rhs(t, y, p_x, p_y),
            t_span,
            [self.psi_initial[0], self.psi_initial[1]],
            t_eval=np.linspace(t_span[0], t_span[1], num_points),
            method='RK45'
        )
        
        return sol.t, np.array([sol.y[0], sol.y[1]])
    
    def einstein_field_equation(self, r):
        """
        Implementación simplificada de la ecuación de campo de Einstein
        para modelar la métrica del agujero de gusano
        """
        # Constantes relevantes para la ecuación de Einstein
        k = 8 * np.pi * G / c**4
        
        # Densidad de energía del agujero de gusano (simplificación)
        rho = self.E / (4/3 * np.pi * r**3)
        
        # Componente temporal del tensor de Ricci (simplificación)
        R_00 = -k * (rho + 3 * self.p(r))
        
        return R_00
    
    def p(self, r):
        """Presión en función de la distancia (simplificación)"""
        return -self.E / (4 * np.pi * r**3 * c**2)
    
    def wormhole_metric(self, r_values):
        """Cálculo simplificado de la métrica del agujero de gusano"""
        # Modelo de métrica de Morris-Thorne simplificado
        b = 2 * G * self.m / c**2  # Radio de garganta simplificado
        
        # Factor de forma del agujero de gusano
        phi = np.zeros_like(r_values)
        g_tt = -np.exp(2 * phi)
        g_rr = 1 / (1 - b / r_values)
        
        return g_tt, g_rr
    
    def quantum_teleportation_circuit(self):
        """
        Crea un circuito cuántico que representa conceptualmente
        la teletransportación a través del agujero de gusano
        """
        # Crear un circuito de 3 qubits
        qc = QuantumCircuit(3, 2)
        
        # Preparar el estado a teleportar (qubit 0)
        qc.h(0)  # Estado superposición
        
        # Crear entrelazamiento entre los extremos del agujero de gusano (qubits 1 y 2)
        qc.h(1)
        qc.cx(1, 2)
        
        # Interacción del estado con la entrada del agujero de gusano
        qc.cx(0, 1)
        qc.h(0)
        
        # Medición (colapso del espacio-tiempo en el modelo conceptual)
        qc.measure([0, 1], [0, 1])
        
        # Correcciones basadas en las mediciones (salida del agujero de gusano)
        qc.cx(1, 2)
        qc.cz(0, 2)
        
        return qc
    
    def run_simulation(self):
        """Ejecuta la simulación completa del modelo"""
        results = {}
        
        # 1. Evolución de la función de onda del fermión de Majorana
        t_span = (0, self.t_teleport)
        t_values, psi_values = self.schrodinger_evolution(t_span)
        results['wave_function'] = {
            't': t_values,
            'psi': psi_values,
            'probability': np.abs(psi_values)**2
        }
        
        # 2. Cálculo de la métrica del agujero de gusano
        r_values = np.linspace(self.d/100, self.d, 1000)
        g_tt, g_rr = self.wormhole_metric(r_values)
        results['metric'] = {
            'r': r_values,
            'g_tt': g_tt,
            'g_rr': g_rr
        }
        
        # 3. Simular el circuito cuántico de teletransportación
        qc = self.quantum_teleportation_circuit()
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(qc, simulator)
        state_vector = job.result().get_statevector()
        results['quantum_circuit'] = {
            'circuit': qc,
            'state_vector': state_vector
        }
        
        return results
    
    def visualize_results(self, results):
        """Visualizar los resultados de la simulación"""
        # Configurar el estilo de las gráficas
        plt.style.use('seaborn-darkgrid')
        
        # Crear una figura con tres subfiguras
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. Evolución de la función de onda
        axs[0].plot(results['wave_function']['t'], 
                   np.abs(results['wave_function']['psi'][0])**2, 
                   label='|ψ₁|²')
        axs[0].plot(results['wave_function']['t'], 
                   np.abs(results['wave_function']['psi'][1])**2, 
                   label='|ψ₂|²')
        axs[0].set_xlabel('Tiempo (s)')
        axs[0].set_ylabel('Probabilidad')
        axs[0].set_title('Evolución de la función de onda del fermión de Majorana')
        axs[0].legend()
        
        # 2. Métrica del agujero de gusano
        axs[1].plot(results['metric']['r'], results['metric']['g_tt'], 
                   label='g₍ₜₜ₎')
        axs[1].plot(results['metric']['r'], results['metric']['g_rr'], 
                   label='g₍ᵣᵣ₎')
        axs[1].set_xlabel('Distancia radial (m)')
        axs[1].set_ylabel('Componentes de la métrica')
        axs[1].set_title('Métrica del agujero de gusano')
        axs[1].legend()
        
        # 3. Representación de las probabilidades finales
        probs = np.abs(results['quantum_circuit']['state_vector'])**2
        states = [f"|{i:03b}⟩" for i in range(8)]
        axs[2].bar(states, probs)
        axs[2].set_xlabel('Estados')
        axs[2].set_ylabel('Probabilidad')
        axs[2].set_title('Distribución de probabilidad después de la teletransportación')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

# Ejemplo de uso del modelo
def run_wormhole_teleportation_simulation():
    # Crear instancia del modelo con los parámetros físicamente realistas
    model = QuantumWormholeModel(
        majorana_mass=2.5e-25,  # kg (escala de masa del neutrino)
        dark_energy=-1.2e-10,   # J (basada en densidad de energía oscura)
        wormhole_distance=1.6e-35, # m (longitud de Planck)
        teleport_time=5.4e-44     # s (tiempo de Planck)
    )
    
    # Ejecutar la simulación
    print("Iniciando simulación del modelo de teletransportación cuántica...")
    results = model.run_simulation()
    
    # Visualizar resultados
    fig = model.visualize_results(results)
    
    print("Simulación completada.")
    return model, results, fig

if __name__ == "__main__":
    model, results, fig = run_wormhole_teleportation_simulation()
    plt.show()
    
    # Mostrar detalles del circuito cuántico
    print("\nCircuito de teletransportación cuántica:")
    print(results['quantum_circuit']['circuit'])
# Establecer límites
        max_g_rr = max([max(m['g_rr']) for m in metrics if not np.any(np.isinf(m['g_rr']))])
        min_g_tt = min([min(m['g_tt']) for m in metrics])
        
        ax.set_xlim(min(r_values), max(r_values))
        ax.set_ylim(min(min_g_tt * 1.1, -max_g_rr * 0.1), max_g_rr * 0.1)
        
        # Función de inicialización
        def init():
            line_tt.set_data([], [])
            line_rr.set_data([], [])
            line_fl.set_data([], [])
            time_text.set_text('')
            return line_tt, line_rr, line_fl, time_text
        
        # Función de animación
        def animate(i):
            metric = metrics[i]
            
            # Limitar los valores extremos para mejor visualización
            g_rr_plot = np.clip(metric['g_rr'], -max_g_rr*0.1, max_g_rr*0.1)
            
            line_tt.set_data(r_values, metric['g_tt'])
            line_rr.set_data(r_values, g_rr_plot)
            line_fl.set_data(r_values, metric['fluctuations'])
            
            time_value = time_points[i]
            time_text.set_text(f'Tiempo: {time_value:.2e} s')
            
            return line_tt, line_rr, line_fl, time_text
        
        # Crear la animación
        anim = FuncAnimation(fig, animate, frames=len(time_points),
                             init_func=init, blit=True, interval=200)
        
        return anim
    
    def create_3d_wormhole_animation(self, results):
        """
        Crear una animación 3D del agujero de gusano que muestre
        las fluctuaciones cuánticas del espacio-tiempo
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Datos para la animación
        time_points = results['metric_evolution']['time_points']
        
        # Parametrización base de la superficie del agujero de gusano
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(-np.pi, np.pi, 40)
        u, v = np.meshgrid(u, v)
        
        # Radio del tubo y distancia al centro
        tube_radius = self.d / 5
        torus_radius = self.d
        
        # Superficie inicial (será actualizada en cada frame)
        x = (torus_radius + tube_radius * np.cos(v)) * np.cos(u)
        y = (torus_radius + tube_radius * np.cos(v)) * np.sin(u)
        z = tube_radius * np.sin(v)
        
        # Crear la superficie inicial
        surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8,
                              rstride=1, cstride=1, linewidth=0,
                              antialiased=True)
        
        # Configurar la visualización 3D
        ax.set_title('Evolución temporal del agujero de gusano cuántico')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_box_aspect([1,1,1])
        
        # Texto para mostrar el tiempo
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
        
        # Función de inicialización
        def init():
            surf.remove()
            time_text.set_text('')
            return surf, time_text
        
        # Función de animación
        def animate(i):
            # Eliminar la superficie anterior
            surf.remove()
            
            # Tiempo actual
            t = time_points[i]
            
            # Calcular las fluctuaciones para este tiempo
            fluctuation_amplitude = np.zeros_like(u)
            
            for j in range(u.shape[0]):
                for k in range(u.shape[1]):
                    # Calcular la distancia radial aproximada
                    r_approx = np.sqrt(x[j,k]**2 + y[j,k]**2 + z[j,k]**2)
                    fluctuation_amplitude[j,k] = self.quantum_fluctuation(t, r_approx/10) * 10
            
            # Crear la nueva geometría con fluctuaciones
            x_new = (torus_radius + tube_radius * np.cos(v)) * np.cos(u)
            y_new = (torus_radius + tube_radius * np.cos(v)) * np.sin(u)
            z_new = tube_radius * np.sin(v)
            
            # Aplicar las fluctuaciones
            x_new += fluctuation_amplitude * np.cos(u) * np.cos(v)
            y_new += fluctuation_amplitude * np.sin(u) * np.cos(v)
            z_new += fluctuation_amplitude * np.sin(v)
            
            # Añadir la nueva superficie
            nonlocal surf
            surf = ax.plot_surface(x_new, y_new, z_new, cmap='inferno', alpha=0.8,
                                 rstride=1, cstride=1, linewidth=0,
                                 antialiased=True, vmin=-0.01, vmax=0.01)
            
            # Actualizar el texto del tiempo
            time_text.set_text(f'Tiempo: {t:.2e} s')
            
            return surf, time_text
        
        # Crear la animación
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=len(time_points), blit=False, interval=200)
        
        return anim

    def compare_majorana_dirac(self):
        """
        Compara las propiedades de fermiones de Majorana vs fermiones de Dirac
        en el contexto de la teletransportación a través de agujeros de gusano
        """
        # Parámetros para la comparación
        t_span = (0, self.t_teleport)
        t_values = np.linspace(t_span[0], t_span[1], 200)
        
        # Evolución para el fermión de Majorana
        _, psi_majorana = self.schrodinger_evolution(t_span, num_points=200)
        
        # Evolución para un fermión de Dirac (duplicando la masa)
        original_mass = self.m
        self.m = original_mass * 2
        _, psi_dirac = self.schrodinger_evolution(t_span, num_points=200)
        self.m = original_mass  # Restauramos el valor original
        
        # Calcular probabilidades de localización
        prob_majorana = np.sum(np.abs(psi_majorana)**2, axis=0)
        prob_dirac = np.sum(np.abs(psi_dirac)**2, axis=0)
        
        # Crear la visualización
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Probabilidad para el fermión de Majorana
        axs[0].plot(t_values, prob_majorana, color='blue', label='Probabilidad total')
        for i in range(4):
            axs[0].plot(t_values, np.abs(psi_majorana[i])**2, '--', alpha=0.6, 
                      label=f'Componente {i+1}')
        
        axs[0].set_ylabel('Probabilidad')
        axs[0].set_title('Fermión de Majorana en el agujero de gusano')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Probabilidad para el fermión de Dirac
        axs[1].plot(t_values, prob_dirac, color='red', label='Probabilidad total')
        for i in range(4):
            axs[1].plot(t_values, np.abs(psi_dirac[i])**2, '--', alpha=0.6, 
                      label=f'Componente {i+1}')
        
        axs[1].set_xlabel('Tiempo (s)')
        axs[1].set_ylabel('Probabilidad')
        axs[1].set_title('Fermión de Dirac en el agujero de gusano')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig

    def quantum_phase_analysis(self):
        """
        Analiza la evolución de las fases cuánticas durante la teletransportación
        """
        # Parámetros para el análisis
        t_span = (0, self.t_teleport)
        t_values, psi_values = self.schrodinger_evolution(t_span, num_points=200)
        
        # Extraer fases
        phases = np.angle(psi_values)
        
        # Calcular diferencia de fase entre componentes
        phase_diff_12 = np.mod(phases[0] - phases[1], 2*np.pi)
        phase_diff_13 = np.mod(phases[0] - phases[2], 2*np.pi)
        phase_diff_14 = np.mod(phases[0] - phases[3], 2*np.pi)
        
        # Calcular entropía de entrelazamiento (aproximación)
        amplitudes = np.abs(psi_values)
        prob_densities = amplitudes**2
        
        # Probabilidades normalizadas para calcular la entropía
        prob_norm = prob_densities / np.sum(prob_densities, axis=0)
        
        # Evitar log(0)
        prob_norm = np.clip(prob_norm, 1e-10, 1.0)
        
        # Entropía de von Neumann (aproximación)
        entropy = -np.sum(prob_norm * np.log2(prob_norm), axis=0)
        
        # Crear la visualización
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Graficar diferencias de fase
        axs[0].plot(t_values, phase_diff_12, label='Φ₁ - Φ₂')
        axs[0].plot(t_values, phase_diff_13, label='Φ₁ - Φ₃')
        axs[0].plot(t_values, phase_diff_14, label='Φ₁ - Φ₄')
        
        axs[0].set_ylabel('Diferencia de fase (rad)')
        axs[0].set_title('Evolución de diferencias de fase cuántica')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Graficar entropía de entrelazamiento
        axs[1].plot(t_values, entropy, color='purple')
        axs[1].set_xlabel('Tiempo (s)')
        axs[1].set_ylabel('Entropía (bits)')
        axs[1].set_title('Entropía de entrelazamiento durante la teletransportación')
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig

def main():
    """Función principal para ejecutar el modelo mejorado"""
    # Crear instancia del modelo
    model = EnhancedQuantumWormholeModel()
    
    # Ejecutar simulación
    results = model.run_enhanced_simulation()
    
    # Visualizar resultados
    fig1, fig2 = model.visualize_enhanced_results(results)
    
    # Crear animación de la evolución del agujero de gusano
    anim = model.create_wormhole_animation(results)
    
    # Crear animación 3D
    anim3d = model.create_3d_wormhole_animation(results)
    
    # Análisis adicionales
    fig_compare = model.compare_majorana_dirac()
    fig_phase = model.quantum_phase_analysis()
    
    # Mostrar visualizaciones
    plt.show()
    
    # Para guardar los resultados
    # fig1.savefig('quantum_wormhole_results.png', dpi=300)
    # fig2.savefig('quantum_wormhole_3d.png', dpi=300)
    # fig_compare.savefig('majorana_dirac_comparison.png', dpi=300)
    # fig_phase.savefig('quantum_phase_analysis.png', dpi=300)
    
    # Para guardar animaciones
    # anim.save('wormhole_evolution.mp4', writer='ffmpeg', dpi=200)
    # anim3d.save('wormhole_3d_evolution.mp4', writer='ffmpeg', dpi=200)
    
    print("Simulación completada con éxito.")
    return results

if __name__ == "__main__":
    main()