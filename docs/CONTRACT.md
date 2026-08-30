# The contract

The behaviours below are **frozen**. Other media software reads the folders this
application writes, so changing any of them changes what that software sees.
Each one names the test that fails if it is broken — those test names are the
real documentation, and they are written as English sentences on purpose.

Run the whole list yourself:

```bash
pytest -v                                     # 260 tests, no GPU needed, ~5s
docker compose run --rm subwright --self-test # 23 checks, inside the image
```

The self-test is the stronger evidence of the two: it runs the real pipeline
against a temporary folder inside the shipped container, so it catches "the
tests pass but the image is broken", which unit tests cannot.

---

## Folder layout

For an ingested video `Foo.mkv`, with `<output>` from the folder's settings:

```
<output>/Foo/
    Foo.mkv          the video, moved here out of the drop folder
    Foo.srt          subtitles
    .translated      success marker, written last
```

| Behaviour | Test |
|---|---|
| Exactly that layout, and nothing else | `test_ingest_produces_exactly_the_expected_layout` |
| A name collision appends `_YYYYmmdd_HHMMSS` rather than merging | `test_ingest_collision_appends_timestamp` |
| The drop folder is left empty | `test_ingest_empties_the_ingest_directory` |
| Files are owned `1000:1000` so media servers can read them | `test_set_owner_never_raises_when_not_permitted` |
| An installation with no folders configured keeps the original single-folder layout | `test_an_installation_with_no_rules_gets_the_original_layout` |
| ...and puts output exactly where it always went | `test_the_default_rule_puts_output_exactly_where_it_always_went` |

**All path construction lives in `layout.py`.** Nothing else builds a path. That
is what makes this list checkable by reading one file.

## Never touching what it did not create

The single most important property here. The watch tree already contains a media
library that Plex and Stash read.

| Behaviour | Test |
|---|---|
| Resume adopts a folder **only** if it holds a `.processing` claim this app wrote | `test_preexisting_library_folder_without_claim_is_untouched` |
| A folder is eligible only because of the claim, nothing else | `test_resume_finds_a_folder_with_a_claim_and_no_translated_marker` |
| A claim left behind after success is not re-run | `test_claim_left_behind_after_success_is_not_resumed` |
| A finished folder is not resumable | `test_completed_folder_is_not_resumed` |
| One folder's interrupted job is never adopted by another folder | `test_resume_only_looks_inside_the_folder_that_owns_it` |
| A disabled folder is left completely alone | `test_a_disabled_folder_is_not_watched` |

The obvious alternative — "any folder with a video and no `.translated`" — would
re-transcribe the entire existing library and overwrite its subtitles.

## Nothing is lost, ever

| Behaviour | Test |
|---|---|
| The video is moved out of the drop folder **before** transcription starts | `test_video_is_moved_before_transcription_starts` |
| A failure leaves no partial `.srt` | `test_failed_ingest_leaves_no_partial_srt` |
| A failed write leaves the original untouched | `test_failed_write_leaves_the_original_untouched` |
| An interrupted job is finished on the next start | `test_interrupted_job_is_found_and_completed_on_restart` |
| A caught failure drops the claim, so a broken file is not retried forever | `test_failed_ingest_drops_the_claim_so_it_is_not_retried_forever` |
| Reprocessing never moves the video | `test_reprocess_never_moves_the_video` |
| Existing subtitles stay readable while they are regenerated | `test_the_old_subtitles_stay_readable_while_a_redo_is_running` |
| A failed redo leaves the originals and no stray backup | `test_a_failed_redo_leaves_the_original_subtitles_and_no_stray_backup` |
| Old subtitles are backed up before being replaced | `test_reprocess_backs_up_existing_subtitles` |

**A caught failure drops the claim; a kill keeps it.** That distinction is the
whole point of the claim file: "this file is broken" and "we died" need opposite
responses.

## Subtitle format

| Behaviour | Test |
|---|---|
| Timestamps are `HH:MM:SS,mmm` with milliseconds **truncated**, not rounded | `test_timestamp_truncates_milliseconds_rather_than_rounding` |
| No cue exceeds 5 seconds | `test_cue_longer_than_five_seconds_is_capped` |
| Numbering is contiguous from 1 after empty cues are dropped | `test_numbering_is_contiguous_when_empty_cues_are_dropped` |
| Byte-for-byte match against output from the original script | `test_renders_golden_file_byte_for_byte` |

Truncation is deliberate. Rounding would shift every timestamp in every
previously generated subtitle by up to a millisecond, making old and new output
incomparable for no benefit.

## Transcription

| Behaviour | Test |
|---|---|
| Output is always English; there is no target-language setting | `test_output_is_always_english_and_there_is_no_target_language_setting` |
| `large-v3-turbo` is never offered — it cannot translate | `test_turbo_model_is_not_offered_because_it_cannot_translate` |
| Voice detection is never disabled by any profile | `test_voice_detection_is_never_switched_off_entirely` |
| Each profile is strictly more permissive than the last | `test_each_profile_is_more_permissive_than_the_last` |
| Standard no longer discards more than stock Whisper | `test_the_standard_profile_no_longer_discards_more_than_whisper_would` |
| An unknown profile falls back rather than failing a job | `test_an_unknown_profile_falls_back_instead_of_failing` |
| `int8` and `cuda` are the defaults, suited to a Pascal card | `test_gpu_defaults_suit_a_pascal_card` |

## Reusing subtitles that already exist

Reuse is an optimisation and must never turn a working file into a failed one.

| Never reused | Test |
|---|---|
| Picture-based tracks (PGS, DVD, DVB) — they need OCR | `test_a_picture_based_track_is_not_extracted` |
| Forced tracks — a few lines, not the film | `test_a_forced_track_is_not_used_as_the_whole_subtitle` |
| Non-English tracks — that is the input, not the answer | `test_a_non_english_track_is_not_used` |
| An empty sidecar | `test_an_empty_sidecar_is_ignored_and_the_video_is_transcribed` |
| A sidecar that is not really a subtitle file | `test_a_sidecar_that_is_not_actually_a_subtitle_file_is_ignored` |
| A sidecar belonging to a different video | `test_a_sidecar_belonging_to_a_different_video_is_ignored` |

| Always falls back to transcribing | Test |
|---|---|
| Extraction failed | `test_a_failed_extraction_falls_back_to_transcribing` |
| Extraction produced nonsense | `test_an_extraction_that_produces_nonsense_falls_back_to_transcribing` |
| No prober available | `test_no_prober_means_embedded_tracks_are_simply_never_looked_for` |
| Reprocess is asked to redo something | `test_reprocessing_regenerates_rather_than_reusing_what_is_there` |

## Settings

| Behaviour | Test |
|---|---|
| Precedence is CLI flag > stored > environment > default | `test_cli_flag_beats_stored_settings` |
| A blank environment variable is ignored, not treated as empty | `test_blank_environment_variable_is_ignored` |
| An invalid value is rejected before it can break the next startup | `test_unknown_stored_model_is_rejected` |
| A misspelt language is caught at save time, not an hour into a job | `test_a_misspelt_language_is_rejected_before_any_gpu_time_is_spent` |
| Saving settings never silently changes the language | `test_saving_settings_does_not_change_the_language` |
| Two folders cannot watch the same directory | `test_two_folders_watching_the_same_directory_are_rejected` |
| A folder the container cannot reach is refused with a reason | `test_a_folder_the_container_cannot_reach_is_refused_with_a_useful_message` |
| A database written by an older version still opens, keeping its history | `test_a_database_written_before_language_was_recorded_still_opens` |

## What no test can promise

**Translation quality.** Nothing here asserts that Whisper's English is any
good. The tests guarantee the tuning is what it claims to be and the plumbing is
correct; whether a given file reads well is judged by eye. Turn on reports
(`SW_WRITE_REPORTS=true`) to see how much audio was heard and how confident the
model was — those are measurements, not judgements.

**That a more permissive profile is better.** It hears more *and* invents more.
There is no setting that only finds true positives.
