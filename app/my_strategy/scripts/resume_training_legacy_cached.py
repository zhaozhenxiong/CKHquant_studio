"""Continue a legacy frozen run with verified completed reference assets."""
from my_strategy.scripts.resume_training_fast import main
from my_strategy.services.czsc_training_legacy_cache import resume_legacy_cached


if __name__ == '__main__':
    main(resume_function=resume_legacy_cached)
