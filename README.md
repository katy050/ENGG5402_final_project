# UR5e touch-navigation trials

The project uses one trial per navigation strategy. The physical runner performs one trial per invocation. Its dry-run mode prints the first planned action without connecting. Run the simulator from the project directory with `python run_both_simulations.py`.

## Commission the experiment

The probe TCP and payload must already be configured on the pendant. Configure the controller safety limits and planes for your lab’s risk assessment. Verify the entire probing plane, probe clearance, joint configurations, and contact stopping behavior with the responsible robot operator. The Python force monitor is an experimental detector. The controller’s configured safety functions provide the independent protection.

Hardware behavior has been tested offline with simulated inputs. The physical runner uses asynchronous **MoveL** and **StopL** through **ur_rtde**. See the [UR5e user manual](https://www.universal-robots.com/manuals/EN/PDF/SW5_26/user-manual-UR5e-PDF_online/710-965-00_UR5e_User_Manual_en_Global.pdf) and the [ur_rtde asynchronous MoveL example](https://sdurobotics.gitlab.io/ur_rtde/pages/examples/basic_motion/move_async_example.html).

Install the dependencies from the project directory.

```bash
python -m pip install -r requirements.txt
python ur5e_runner.py --scenario unknown --planner MDP --step 5
```

Copy `robot_settings.example.json` to `robot_settings.json` and enter the values validated for your probe and installation. The template contains no motion or force defaults. Capture the table frame in the robot’s base coordinates with `--teach`. Teach the three cell centers at the operating TCP height and orientation. This mode uses receive-only RTDE.

```bash
python ur5e_runner.py --teach --ip ROBOT_IP
```

For the current layout, place B1–B4 at actual lower-left coordinates **(8, 19)**, **(23, 10)**, **(37, 19)**, and **(46, 13)** cm, respectively. These positions are for the operator. Use the same fixed layout for both experiments. The software uses the rough probability map in the rough experiment and a uniform prior in the unknown experiment.

## Run a trial

Manually place the stationary, unloaded probe at cell **(20, 3)**, in row-column order, at the taught height and orientation before each invocation. The goal is **(20, 56)**. Log names must be unique so earlier trials are preserved.

```bash
python ur5e_runner.py --execute --ip ROBOT_IP --scenario rough --planner 'A*' --log rough_astar_01.jsonl
python ur5e_runner.py --execute --ip ROBOT_IP --scenario unknown --planner MDP --step 5 --log unknown_mdp_5cm_01.jsonl
```

Use planner `'A*'` or `MDP` in either scenario. The unknown scenario accepts `--step 1`, `3`, or `5`. After a detected contact and retreat, the probe travels parallel to the approached
box face in the selected increment, then makes one guarded 1 cm check in the
original approach direction. A hit during that check causes a retreat followed
by another parallel step. A clear check completes the 1 cm move and returns
control immediately to the planner. This face-following mode assumes
grid-aligned box faces. A collision on a parallel step or a workspace edge
can trigger the same one-time direction reversal as the simulator. Repeat from the start with a new log filename for each trial.

The log records the phase of each commanded move (`navigate`, `edge_follow`,
or `edge_check`), live force, TCP pose, estimated obstacle cells, contacts,
completed distance, total distance, and elapsed wall-clock time. Contact cells are estimated from measured TCP position and, for diagonal moves, force direction. Clear moves mark traversed center-line cells free. Physical localization and controller stopping behavior can differ from the ideal grid simulation.

## Generate the simulation plots

From the project directory, run one trial for each of the eight navigation strategies.

```bash
python run_both_simulations.py
```

The script saves the maps, individual paths, and a combined view of the
eight paths in `figures/`. It also writes `strategy_metrics.pdf` to compare
total distance traveled (including retreats), simulated time, and obstacle
contacts for all eight strategies. It saves `time_by_strategy.png` with the
timing comparison and `path_legend.png` with the path symbols. Results are written
to `csv/trial_results.csv` and `csv/averages.csv`. With one trial per strategy,
each average equals the corresponding trial value.

Edit `plot_settings.py` to change shared Matplotlib defaults or adjust the
size, fonts, spacing, and grid for an individual figure or the GIF replay.

Add `--animate` to save eight individual GIFs and the combined
`simulation_animation.gif` in `animation/`. The GIFs replay the first trial
at 12× speed and show the probe's route, contacts and retreats, edge-following
moves, and clear checks.

```bash
python run_both_simulations.py --animate
```

## Build the report

Compile `project_report.tex` from the project directory with PDFLaTeX and
BibTeX. For Overleaf, upload `project_report.tex`, `references.bib`, and the
`figures/` directory, then select `project_report.tex` as the main document.
