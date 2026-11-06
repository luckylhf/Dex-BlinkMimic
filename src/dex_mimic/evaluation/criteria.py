"""Evaluation stops and tracking quality are separate from physical fall detection."""

DEFAULT_PROFILE = 'jump_tracking_v2'
PROFILES = {
    'strict_v1': {'height_stop_enabled': True},
    'height_diagnostic_v1': {'height_stop_enabled': False},
    'jump_tracking_v2': {'height_stop_enabled': False},
}


def assess(height_error, tilt_error, term_height, term_tilt, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown evaluation termination profile: ' + profile)
    height = bool(height_error > term_height)
    tilt = bool(tilt_error > term_tilt)
    stopped = tilt or (PROFILES[profile]['height_stop_enabled'] and height)
    return dict(height_violation=height, tilt_violation=tilt,
                tracking_violation=height or tilt, failed=stopped,
                termination_reason=('tilt_tracking_deviation' if tilt else
                                    'height_tracking_deviation' if stopped else None))


def summarize(rows, expected_steps, control_dt):
    stopped = any(row['failed'] for row in rows)
    completed = len(rows) == expected_steps and not stopped
    tracking_passed = not any(row['tracking_violation'] for row in rows)
    reason = rows[-1].get('termination_reason') if stopped and rows else None
    reason = reason or ('reference_end' if completed else 'step_limit')
    outcome = ('tilt_tracking_termination' if reason == 'tilt_tracking_deviation' else
               'height_tracking_termination' if reason == 'height_tracking_deviation' else
               'completed_with_tracking_deviation' if completed and not tracking_passed else
               'completed' if completed else 'step_limit')
    return dict(completed=completed, failed=stopped, termination_reason=reason,
                outcome=outcome, tracking_criteria_passed=tracking_passed,
                # Rzz tracking error is a proxy, not a contact-based fall detector.
                physical_fall_detected=None,
                height_violation_frames=sum(row['height_violation'] for row in rows),
                height_violation_seconds=sum(row['height_violation'] for row in rows)*control_dt,
                max_height_error_m=max((row['height_error_m'] for row in rows), default=0.))
