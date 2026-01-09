mod app;
mod collectors;
mod ui;

use anyhow::Result;
use app::{App, KillState};
use collectors::Collectors;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, stdout};
use std::time::{Duration, Instant};

const UPDATE_INTERVAL_MS: u64 = 500;

fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let result = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::default();
    let mut collectors = Collectors::new();

    let interval = Duration::from_millis(UPDATE_INTERVAL_MS);
    let interval_secs = UPDATE_INTERVAL_MS as f64 / 1000.0;
    let mut last_update = Instant::now() - interval; // Force immediate first update

    loop {
        // Check if we need to update data
        if last_update.elapsed() >= interval {
            collectors.collect(&mut app, interval_secs);
            last_update = Instant::now();
        }

        // Draw UI
        terminal.draw(|frame| ui::draw(frame, &app))?;

        // Handle input with timeout to maintain update rate
        let timeout = interval.saturating_sub(last_update.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.kill_state {
                        KillState::None => match key.code {
                            KeyCode::Char('q') | KeyCode::Char('Q') => {
                                app.should_quit = true;
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                if app.selected_process > 0 {
                                    app.selected_process -= 1;
                                }
                            }
                            KeyCode::Down | KeyCode::Char('j') => {
                                if app.selected_process + 1 < app.processes.len() {
                                    app.selected_process += 1;
                                }
                            }
                            KeyCode::Left | KeyCode::Char('h') => {
                                app.sort_column = app.sort_column.prev();
                                app.sort_processes();
                            }
                            KeyCode::Right | KeyCode::Char('l') => {
                                app.sort_column = app.sort_column.next();
                                app.sort_processes();
                            }
                            KeyCode::Char('K') => {
                                // Capture PID immediately to prevent race condition
                                if let Some(pid) = app.selected_pid() {
                                    app.kill_state = KillState::Confirming(pid);
                                }
                            }
                            _ => {}
                        },
                        KillState::Confirming(pid) => match key.code {
                            KeyCode::Enter => {
                                // Kill using the stored PID
                                kill_process(pid);
                                app.kill_state = KillState::None;
                            }
                            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                                app.kill_state = KillState::None;
                            }
                            _ => {}
                        },
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn kill_process(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    #[cfg(windows)]
    {
        // Windows process termination would go here
        let _ = pid;
    }
}
