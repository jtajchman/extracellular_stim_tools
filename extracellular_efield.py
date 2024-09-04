import matplotlib.pyplot as plt
import numpy as np
from math import pi, ceil
from .units import *

class Shape:
    def __init__(
        self,
        shape: str,
        pulse_width_ms: float,
    ):
        # from sympy import Symbol, Function, diff, lambdify, sin
        import sympy as sp

        # Shape can be "Ideal_Sine", "Ideal_Square", "Biphasic", "Half-Biphasic", "Monophasic"
        # Only "Ideal_Sine" and "Ideal_Square" are supported currently
        self.shape = shape
        self.pulse_width_ms = pulse_width_ms
        if self.shape not in ["Ideal_Sine", "Ideal_Square"]:
            raise ValueError(
                f"Pulse shape [{self.shape}] must be \"Ideal_Sine\" or \"Ideal_Square\""
            )
        if self.pulse_width_ms <= 0:
            raise ValueError(f"pulse_width_ms [{self.pulse_width_ms}] must be > 0")
        t = sp.Symbol("t")
        if self.shape == "Ideal_Sine":
            sin_freq_kHz = 2 * pi / pulse_width_ms # Angular frequency of sine wave
            efield_waveform = sp.cos(t * sin_freq_kHz) # Sinusoidal TMS pulses have cosinusoidal E-field waveforms
        elif self.shape == "Ideal_Square":
            efield_waveform = 1 # Constant value of 1
        self.efield_waveform_function = sp.lambdify(t, efield_waveform) # Define electric field waveform shape function (amplitude of 1; scaling comes later)


class Pattern:
    def __init__(
        self,
        pulse_shape: Shape,  # Shape of the pulse
        pattern: str | None = None,  # Pattern of the burst
        num_tms_pulses_per_burst: int | None = None,  # Number of pulses in a burst
        pulse_interval_within_burst_ms: float | None = None,  # Duration of interval between pulses in a burst
        pulse_onset_interval_within_burst_ms: float | None = None,  # Duration of interval between onset of pulses in a burst
        pulse_freq_within_burst_Hz: float | None = None,  # Frequency of pulse onsets in a burst
    ):
        """If pattern is "Single" then num_tms_pulses_per_burst set to 1 (and pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, & pulse_freq_within_burst_Hz are meaningless)
        If "TBS" then num_tms_pulses_per_burst set to 3 (Theta burst stimulation)

        pulse_interval_within_burst_ms = pulse_onset_interval_within_burst_ms - pulse_shape.pulse_width_ms
        pulse_onset_interval_within_burst_ms = 1/pulse_freq_within_burst_Hz
        Highest priority when defined | pulse_interval_within_burst_ms > pulse_onset_interval_within_burst_ms > pulse_freq_within_burst_Hz | lowest priority"""

        # Set attributes
        self.pulse_shape = pulse_shape
        self.pattern = pattern
        pulse_width_ms = self.pulse_shape.pulse_width_ms

        # Implement pattern
        if self.pattern == "Single":
            num_tms_pulses_per_burst = 1
        elif self.pattern == "TBS":
            # An interval parameter must still be defined
            num_tms_pulses_per_burst = 3
        
        # Check that num_tms_pulses_per_burst and pattern are valid
        if type(num_tms_pulses_per_burst) != int or num_tms_pulses_per_burst < 1:
            raise ValueError(
                f"num_tms_pulses_per_burst [{num_tms_pulses_per_burst}] must be defined as a positive non-zero integer or pulse pattern must be categorized as either \"Single\" or \"TBS\""
                )
        self.num_tms_pulses_per_burst = num_tms_pulses_per_burst

        if num_tms_pulses_per_burst == 1:
            pulse_interval_within_burst_ms = 0 # Meaningless, but marks it as set

        # Check that pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, and pulse_freq_within_burst_Hz are valid
        if pulse_interval_within_burst_ms == None:
            if pulse_onset_interval_within_burst_ms == None:
                if pulse_freq_within_burst_Hz == None:
                    raise ValueError(
                        "pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, or pulse_freq_within_burst_Hz" \
                            f"must be defined if num_tms_pulses_per_burst [{num_tms_pulses_per_burst}] > 1"
                    )
                else:
                    if pulse_freq_within_burst_Hz > 1/pulse_width_ms * kHz: # Comparison in Hz
                        raise ValueError(
                            f"pulse_freq_within_burst_Hz [{pulse_freq_within_burst_Hz}] must be <= 1/(pulse_width_ms) [{1/pulse_width_ms * kHz}]"
                        )
                    pulse_interval_within_burst_ms = 1/pulse_freq_within_burst_Hz * s - pulse_width_ms
            else:
                if pulse_onset_interval_within_burst_ms < pulse_width_ms:
                    raise ValueError(
                        f"pulse_onset_interval_within_burst_ms [{pulse_onset_interval_within_burst_ms}] must be >= pulse_width_ms [{pulse_width_ms}]"
                    )
                pulse_interval_within_burst_ms = pulse_onset_interval_within_burst_ms - pulse_width_ms
        if pulse_interval_within_burst_ms < 0:
            raise ValueError(f"pulse_interval_within_burst_ms [{pulse_interval_within_burst_ms}] must be >= 0")
        
        # Set remaining attributes
        self.pulse_interval_within_burst_ms = pulse_interval_within_burst_ms
        self.pulse_onset_interval_within_burst_ms = self.pulse_interval_within_burst_ms + pulse_width_ms
        self.pulse_freq_within_burst_Hz = 1 / self.pulse_onset_interval_within_burst_ms * kHz


def generate_efield(
    burst_freq_Hz: float | None,  # Frequency of pulse bursts in Hz (meaningless for sTMS & tDCS? (TODO), in which case this is None)
    simulation_duration_ms: float,  # Duration of waveform
    dt: float,  # Duration of time step in ms
    #active_dt: float | None, # Duration of time step in ms when electric field activity is present; defaults to be equivalent to dt TODO
    stim_start_ms: float,  # Initial waiting period
    stim_end_ms: float | None, # Time when stimulation ends
    total_num_tms_pulse_bursts: int | None, # Number of pulse bursts after stim_start_ms
    efield_amplitude_mV_per_um: float,  # Amplitude of the max E-field in the desired waveform
    pat: Pattern,  # Pattern object containing data on the waveform
):
    rd = 9 # Rounding precision to correct floating point error (rounds to 10^-rd ms)
    efield_waveform_function = pat.pulse_shape.efield_waveform_function  # E-field pulse function (SymPy)
    pulse_width_ms = pat.pulse_shape.pulse_width_ms  # Duration of one pulse
    pulse_onset_interval_within_burst_ms = pat.pulse_onset_interval_within_burst_ms # Duration of interval between the onset of pulses in a burst
    inter_p_interval_ms = pat.pulse_interval_within_burst_ms # Duration of interval between pulses within a burst
    num_tms_pulses_per_burst = pat.num_tms_pulses_per_burst # Number of pulses in one burst
    burst_width_ms = (num_tms_pulses_per_burst - 1) * inter_p_interval_ms + num_tms_pulses_per_burst * pulse_width_ms # Duration of one burst of pulses (time of intervals + time of pulses)
    
    if total_num_tms_pulse_bursts != None:
        if total_num_tms_pulse_bursts <= 1:
            burst_freq_Hz = None
        elif burst_freq_Hz == None:
            raise ValueError(f"rtms_pulse_burst_freq_Hz must be defined if total_num_tms_pulse_bursts [{total_num_tms_pulse_bursts}] > 1")
            # Situation only applicable/possible with rTMS
    
    if burst_freq_Hz != None:
        burst_onset_interval_ms = 1 / burst_freq_Hz * s # Duration of interval between the onset of bursts of pulses
        inter_burst_interval_ms = burst_onset_interval_ms - burst_width_ms # Duration of interval between bursts of pulses
    else:
        burst_onset_interval_ms = None
        inter_burst_interval_ms = None
    if dt > pulse_width_ms: # Fixes strange behavior with impractically coarse waveform resolution
        dt = pulse_width_ms # TODO: can remove this when sampling-compatible efield construction is done


    # Check one last potential conflict in parameter values
    if inter_burst_interval_ms != None:
        if inter_burst_interval_ms < 0:
            raise ValueError(
                f"Duration of pulse burst [{burst_width_ms} ms] must be <= interval between pulse burst onset" \
                    f"(1/rtms_pulse_burst_freq_Hz or 1/tacs_freq_Hz) [{burst_onset_interval_ms} ms]"
            )

    # Construct waveform of electric field, taking advantage of linear interpolation between time points

    # Initialize variables for waveform construction
    time = [] # Points in time
    wav = []  # E-field waveform at each point in time

    cur_t = stim_start_ms # Current time as progressing through loop
    pulse_start_times_ms = [] # List of pulse start times
    num_bursts = 0 # Number of bursts accounted for
    while cur_t < stim_end_ms: # Iterate through stim duration and build pulse_start_times
        for pcount in range(num_tms_pulses_per_burst): # Build one burst
            pulse_start_times_ms.append(cur_t)
            if pcount == num_tms_pulses_per_burst-1: # If at the last pulse of the burst
                cur_t += pulse_width_ms # Advance to the end of the burst
            else: # If in the middle of a burst
                cur_t += pulse_onset_interval_within_burst_ms # Advance to the beginning of the next pulse in the burst
        num_bursts += 1
        if total_num_tms_pulse_bursts != None: # If using total_num_tms_pulse_bursts
            if num_bursts >= total_num_tms_pulse_bursts: # And the number of bursts has reached the total limit
                break # End the building process
        cur_t += inter_burst_interval_ms # If not, advance to the start of the next burst

    pulse_end_times_ms = [start_time + pulse_width_ms for start_time in pulse_start_times_ms]

    # if inter_burst_interval_ms != None:
    #     while cur_t < stim_end_ms and num_bursts < total_num_tms_pulse_bursts: # Iterate through duration and build pulse_start_times
    #         for pcount in range(num_tms_pulses_per_burst):
    #             pulse_start_times.append(cur_t)
    #             if pcount == num_tms_pulses_per_burst-1: # If at the end of the burst
    #                 cur_t += inter_burst_interval_ms + pulse_width_ms
    #             else: # If in the middle of a burst
    #                 cur_t += pulse_onset_interval_within_burst_ms
    # else:
    #     for pcount in range(num_tms_pulses_per_burst):
    #         pulse_start_times.append(cur_t)
    #         cur_t += pulse_onset_interval_within_burst_ms

    effective_pulse_time_steps = ceil(pulse_width_ms / dt)
    effective_pulse_width_ms = int(effective_pulse_time_steps * dt) # Effective length of pulse when accounting for sampling resolution
    if pat.pulse_shape.shape == "Ideal_Square":
        npoints_pulse = 2 # Only need the first and last time point of the pulse, as it does not change over the duration
    else:
        npoints_pulse = effective_pulse_time_steps + 1  # Number of time points within a pulse
    pulse_t = np.linspace(0, effective_pulse_width_ms, npoints_pulse) # Time points within a pulse starting at t=0
    print(pulse_t)
    pulse = [efield_waveform_function(t) * efield_amplitude_mV_per_um for t in pulse_t] # Sample points of the pulse waveform; scale by efield_amplitude_mV_per_um
    
    if stim_start_ms > 0:
        # Start of initial silent period
        time.append(0)
        wav.append(0)

    # TODO: write time & wav in a loop so that sampling is represented accurately
    # current_time_ms = 0
    # completed_pulses = 0
    # while current_time_ms < simulation_duration_ms:
    #     # Check whether we are within a pulse
    #     if current_time_ms >= pulse_start_times_ms[completed_pulses] and \
    #         current_time_ms <= pulse_end_times_ms[completed_pulses]:
    #         # Do something
        
    #     # Check whether a pulse has just ended and we are not within another pulse
    #         # Add post-pulse buffer
    #         pass

    pulse_end_time = 0
    for i, pulse_start in enumerate(pulse_start_times_ms):
        # Pre-pulse buffer
        last_silent_t = pulse_start - dt # Last time point of silent period before pulse start
        if last_silent_t > pulse_end_time: # If we've advanced far enough to need to specify the end of the silent period
            # Write end of silent period
            time.append(last_silent_t)
            wav.append(0)

        # Write pulse
        time.extend(pulse_t + pulse_start)
        wav.extend(pulse)

        # Post-pulse buffer
        pulse_end_time = pulse_start + effective_pulse_width_ms # Time of the end of the pulse
        start_silent_t = pulse_end_time + dt # First time point of silent period after pulse end
        if (i == len(pulse_start_times_ms)-1 # If this is the last pulse
            or start_silent_t < pulse_start_times_ms[i+1]): # or if we will advance far enough to need to specify the start of a silent period
            # Write start of silent period
            time.append(start_silent_t)
            wav.append(0)
    # Buffers necessary for simulation time steps when outside of a pulse

    time = [round(t, rd) for t in time] # Correct floating point error

    if time[-1] < simulation_duration_ms: # If the time course does not last the full duration
        # Place a silent period until the end of the simulation
        time.append(simulation_duration_ms)
        wav.append(0)

    if time[-1] > simulation_duration_ms: # If the time course is longer than the full duration
        # Trim the time course to fit the duration to save resources
        # Find index of the last time point less than the duration
        ind_last_t = 0
        for ind, t in reversed(list(enumerate(time))): # Going backwards will probably take less computation
            if t < simulation_duration_ms:
                ind_last_t = ind
                break
        time = time[:ind_last_t+2] # +2 because we want to still include the last point before and the first point after the duration
        wav = wav[:ind_last_t+2]

    return [wav, time]

def check_nonspecific_parameters(
    simulation_duration_ms,
    stim_start_ms,
    stim_end_ms,
    sampling_period_ms
):
    # Check that the parameters which are not specific to stimulation type are valid
    if simulation_duration_ms <= 0:
        raise ValueError(f"simulation_duration_ms [{simulation_duration_ms}] must be > 0")
    if stim_start_ms < 0:
        raise ValueError(f"stim_start_ms [{stim_start_ms}] must be >= 0")
    if simulation_duration_ms <= stim_start_ms:
        raise ValueError(f"simulation_duration_ms [{simulation_duration_ms}] must be > stim_start_ms [{stim_start_ms}]")
    # Also set stim_end_ms if it is not set properly
    if stim_end_ms == None or stim_end_ms > simulation_duration_ms:
        stim_end_ms = simulation_duration_ms
    if stim_end_ms <= stim_start_ms:
        raise ValueError(f"stim_end_ms [{stim_end_ms}] must be > stim_start_ms [{stim_start_ms}]")
    if sampling_period_ms <= 0:
        raise ValueError(f"sampling_period_ms [{sampling_period_ms}] must be > 0")
    return stim_end_ms
    
def plot_efield(wav, time):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("E-field Plots")
    ax[0].step(time, wav, where="post")
    ax[0].set_title("Raw Step Function")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("E-field (mV/um)")

    ax[1].plot(time, wav)
    ax[1].set_title("Interpolated Function")
    ax[1].set_xlabel("Time (ms)")
    #ax[1].set_ylabel("E-field (mV/um)")
    
    dt = 0.025
    num_sample_steps = ceil(time[-1] / dt)
    effective_duration_ms = int(num_sample_steps * dt)
    num_sample_points = num_sample_steps + 1  # Number of time points within the duration
    sampled_time = np.linspace(0, effective_duration_ms, num_sample_points, endpoint=False)
    sampled_time = np.append(sampled_time, time[-1])

    sampled_wav = np.interp(sampled_time, time, wav)

    ax[2].step(sampled_time, sampled_wav, where="post")
    ax[2].set_title(f"Sampled Step Function with dt={dt} ms")
    ax[2].set_xlabel("Time (ms)")
    #ax[2].set_ylabel("E-field (mV/um)")

def get_efield_sTMS(
    simulation_duration_ms: float,
    efield_amplitude_V_per_m: float,
    stim_start_ms: float = 0.,
    sampling_period_ms: float = 1e-3,
    tms_pulse_shape: str = "Ideal_Sine",
    tms_pulse_width_ms: float = 100e-3,
    tms_pulse_burst_pattern: str = "Single",
    num_tms_pulses_per_burst: int | None = None,
    pulse_interval_within_burst_ms: float | None = None,
    pulse_onset_interval_within_burst_ms: float | None = None,
    pulse_freq_within_burst_Hz: float | None = None,
    plot: bool = False,
):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    sampling_period_ms: Temporal resolution of pulses in ms (should be <= within-pulse simulation dt)
    tms_pulse_shape: Qualitative description of TMS waveform (see Shape class)
    tms_pulse_width_ms: Period of TMS pulse in ms
    tms_pulse_burst_pattern: Qualitative description of stimulation pattern (see Pattern class)
    num_tms_pulses_per_burst: Number of pulses in one burst of a pattern
    pulse_interval_within_burst_ms: Duration of interval between pulses in a burst in ms
    pulse_onset_interval_within_burst_ms: Duration of interval between onset of pulses in a burst in ms
    pulse_freq_within_burst_Hz: Frequency of pulse onsets in a burst in Hz
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms = check_nonspecific_parameters(
        simulation_duration_ms=simulation_duration_ms,
        stim_start_ms=stim_start_ms,
        stim_end_ms=None,
        sampling_period_ms=sampling_period_ms
    )



    wav, time = generate_efield(
        burst_freq_Hz=None,
        simulation_duration_ms=simulation_duration_ms,  
        dt=sampling_period_ms, 
        stim_start_ms=stim_start_ms,  
        stim_end_ms=stim_end_ms,
        total_num_tms_pulse_bursts=1,
        efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
        pat=Pattern(
            pulse_shape=Shape(shape=tms_pulse_shape, pulse_width_ms=tms_pulse_width_ms),
            pattern=tms_pulse_burst_pattern,
            num_tms_pulses_per_burst=num_tms_pulses_per_burst,
            pulse_interval_within_burst_ms=pulse_interval_within_burst_ms,
            pulse_onset_interval_within_burst_ms=pulse_onset_interval_within_burst_ms,
            pulse_freq_within_burst_Hz=pulse_freq_within_burst_Hz,
        ),
    )

    if plot:
        plot_efield(wav, time)
    
    return wav, time
    
def get_efield_rTMS(
    simulation_duration_ms: float,
    efield_amplitude_V_per_m: float,
    rtms_pulse_burst_freq_Hz: float,
    stim_start_ms: float = 0.,
    stim_end_ms: float | None = None,
    total_num_tms_pulse_bursts: int | None = None,
    sampling_period_ms: float = 1e-3,
    tms_pulse_shape: str = "Ideal_Sine",
    tms_pulse_width_ms: float = 100e-3,
    tms_pulse_burst_pattern: str = "Single",
    num_tms_pulses_per_burst: int | None = None,
    pulse_interval_within_burst_ms: float | None = None,
    pulse_onset_interval_within_burst_ms: float | None = None,
    pulse_freq_within_burst_Hz: float | None = None,
    plot: bool = False,
):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    total_num_tms_pulse_bursts: Total number of pulse bursts to include in time course
        Either stim_end_ms or total_num_tms_pulse_bursts will determine the number of pulse bursts based on which is more restrictive
    rtms_pulse_burst_freq_Hz: Frequency of rTMS pulse bursts
    sampling_period_ms: Temporal resolution of pulses in ms (should be <= within-pulse simulation dt)
    tms_pulse_shape: Qualitative description of TMS waveform (see Shape class)
    tms_pulse_width_ms: Period of TMS pulse in ms
    tms_pulse_burst_pattern: Qualitative description of stimulation pattern (see Pattern class)
    num_tms_pulses_per_burst: Number of pulses in one burst of a pattern
    pulse_interval_within_burst_ms: Duration of interval between pulses in a burst in ms
    pulse_onset_interval_within_burst_ms: Duration of interval between onset of pulses in a burst in ms
    pulse_freq_within_burst_Hz: Frequency of pulse onsets in a burst in Hz
        Only one of pulse_interval_within_burst_ms, pulse_onset_interval_within_burst_ms, or pulse_freq_within_burst_Hz must be defined
        Highest priority when defined | pulse_interval_within_burst_ms > pulse_onset_interval_within_burst_ms > pulse_freq_within_burst_Hz | lowest priority
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms = check_nonspecific_parameters(
        simulation_duration_ms=simulation_duration_ms,
        stim_start_ms=stim_start_ms,
        stim_end_ms=stim_end_ms,
        sampling_period_ms=sampling_period_ms
    )

    # Check that the rTMS-specific parameters are valid
    if total_num_tms_pulse_bursts != None:
        if total_num_tms_pulse_bursts < 0:
            raise ValueError(f"total_num_tms_pulse_bursts [{total_num_tms_pulse_bursts}] must be >= 0")
    if rtms_pulse_burst_freq_Hz <= 0:
        raise ValueError(f"rtms_pulse_burst_freq_Hz [{rtms_pulse_burst_freq_Hz}] must be > 0")

    wav, time = generate_efield(
        burst_freq_Hz=rtms_pulse_burst_freq_Hz,
        simulation_duration_ms=simulation_duration_ms,  
        dt=sampling_period_ms, 
        stim_start_ms=stim_start_ms,  
        stim_end_ms=stim_end_ms,
        total_num_tms_pulse_bursts=total_num_tms_pulse_bursts,
        efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
        pat=Pattern(
            pulse_shape=Shape(shape=tms_pulse_shape, pulse_width_ms=tms_pulse_width_ms),
            pattern=tms_pulse_burst_pattern,
            num_tms_pulses_per_burst=num_tms_pulses_per_burst,
            pulse_interval_within_burst_ms=pulse_interval_within_burst_ms,
            pulse_onset_interval_within_burst_ms=pulse_onset_interval_within_burst_ms,
            pulse_freq_within_burst_Hz=pulse_freq_within_burst_Hz,
        ),
    )

    if plot:
        plot_efield(wav, time)
    
    return wav, time
    
def get_efield_tACS(
    simulation_duration_ms: float,
    efield_amplitude_V_per_m: float,
    stim_start_ms: float = 0.,
    stim_end_ms: float | None = None,
    sampling_period_ms: float = 25e-3,
    tacs_freq_Hz: float | None = None,
    plot: bool = False,
):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    sampling_period_ms: Temporal resolution of pulses in ms (should be <= simulation dt)
    tacs_freq_Hz: Frequency of tACS stimulation
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms = check_nonspecific_parameters(
        simulation_duration_ms=simulation_duration_ms,
        stim_start_ms=stim_start_ms,
        stim_end_ms=stim_end_ms,
        sampling_period_ms=sampling_period_ms
    )

    # Check that the tACS-specific parameter is valid
    if tacs_freq_Hz <= 0:
        raise ValueError(f"tacs_freq_Hz [{tacs_freq_Hz}] must be > 0")

    wav, time = generate_efield(
        burst_freq_Hz=tacs_freq_Hz,
        simulation_duration_ms=simulation_duration_ms,  
        dt=sampling_period_ms, 
        stim_start_ms=stim_start_ms,  
        stim_end_ms=stim_end_ms,
        total_num_tms_pulse_bursts=None,
        efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
        pat=Pattern(
            pulse_shape=Shape(shape="Ideal_Sine", pulse_width_ms=1/tacs_freq_Hz * s),
            pattern="Single",
        ),
    )

    if plot:
        plot_efield(wav, time)
    
    return wav, time
    
def get_efield_tDCS(
    simulation_duration_ms: float,
    efield_amplitude_V_per_m: float,
    stim_start_ms: float = 0.,
    stim_end_ms: float | None = None,
    sampling_period_ms: float = 1e-3,
    plot: bool = False,
):
    """
    simulation_duration_ms: Duration of simulation in ms
    efield_amplitude_V_per_m: Amplitude of electric field pulse in V/m TODO: Typical values for stim type
    stim_start_ms: Time when stimulation starts in ms
    stim_end_ms: Time when stimulation ends in ms
    sampling_period_ms: Temporal resolution of pulses in ms (should be <= simulation dt) TODO: clarify purpose for tDCS
    plot: Whether to plot the electric field time course
    
    Returns electric field time course in mV/um (or V/mm)

    Returns time course in ms
    """
    stim_end_ms = check_nonspecific_parameters(
        simulation_duration_ms=simulation_duration_ms,
        stim_start_ms=stim_start_ms,
        stim_end_ms=stim_end_ms,
        sampling_period_ms=sampling_period_ms
    )

    wav, time = generate_efield(
        burst_freq_Hz=None,
        simulation_duration_ms=simulation_duration_ms,  
        dt=sampling_period_ms, 
        stim_start_ms=stim_start_ms,  
        stim_end_ms=stim_end_ms,
        total_num_tms_pulse_bursts=1,
        efield_amplitude_mV_per_um=efield_amplitude_V_per_m / (mV/um), # Convert from V/m to mV/um (or V/mm)
        pat=Pattern(
            pulse_shape=Shape(shape="Ideal_Square", pulse_width_ms=stim_end_ms-stim_start_ms),
            pattern="Single",
        ),
    )

    if plot:
        plot_efield(wav, time)
    
    return wav, time