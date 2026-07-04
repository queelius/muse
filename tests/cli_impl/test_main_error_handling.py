"""Regression tests for `muse.cli.main`'s click-exception handling.

`main()` is the entry point the shipped `muse` binary calls (see
`pyproject.toml`'s console_scripts entry), invoking the typer app with
`standalone_mode=False` so `typer.Exit` translates into a plain return
code. The price of `standalone_mode=False` is that click (and typer's
own `_main` override) never call `ClickException.show()` on the
non-standalone path -- rendering is entirely the caller's job. Before
this fix, `main()` caught `click.exceptions.ClickException` and
returned `e.exit_code` without ever calling `e.show()`, so every usage
error (unknown option, missing argument, bad value) exited nonzero with
completely empty stdout+stderr on the real `muse` binary: silent
failures.

Subprocess-level tests (`tests/test_cli.py`) invoke
`python -m muse.cli ...`, which calls `app()` directly in standalone
mode and therefore never exercises this bug: standalone mode does call
`show()` internally. Only `main()` itself needs direct, in-process
coverage.
"""
from __future__ import annotations


def test_missing_required_args_shows_usage_error(capsys):
    """`models set-device` with no id/device is a click UsageError.

    Before the fix this exited 2 with empty stdout AND empty stderr.
    """
    from muse.cli import main

    rc = main(["models", "set-device"])
    captured = capsys.readouterr()

    assert rc != 0
    combined = captured.out + captured.err
    assert combined.strip(), "usage error must render a message, not exit silently"


def test_unknown_option_shows_usage_error(capsys):
    """An unrecognized flag is also a click UsageError (NoSuchOption)."""
    from muse.cli import main

    rc = main(["models", "list", "--this-flag-does-not-exist"])
    captured = capsys.readouterr()

    assert rc != 0
    combined = captured.out + captured.err
    assert combined.strip(), "unknown-option error must render a message"


def test_no_args_still_shows_help_without_double_print(capsys):
    """The pre-existing no-args-is-help path must keep working: help is
    shown exactly once (not duplicated) and the exit code stays nonzero.
    """
    from muse.cli import main

    rc = main([])
    captured = capsys.readouterr()

    assert rc != 0
    # The rich-rendered help (printed as a side effect of building
    # NoArgsIsHelpError's message) must appear, and only once.
    assert captured.out.count("Usage:") == 1
