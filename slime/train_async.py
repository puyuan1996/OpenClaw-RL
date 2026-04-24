import ray
import wandb

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import should_run_periodic_action


def _relay_pending_metrics(result):
    """Log pending wandb metrics relayed from secondary processes."""
    if not result:
        return
    if wandb.run is None:
        return
    metrics_list = result if isinstance(result, list) else [result]
    for item in metrics_list:
        if isinstance(item, list):
            for m in item:
                if isinstance(m, dict):
                    wandb.log(m)
        elif isinstance(item, dict):
            wandb.log(item)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"], pgs.get("prm"))

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            gen_result = ray.get(rollout_data_next_future)
            if isinstance(gen_result, tuple):
                rollout_data_curr_ref, pending = gen_result
                _relay_pending_metrics(pending)
            else:
                rollout_data_curr_ref = gen_result

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref)))
            _relay_pending_metrics(ray.get(critic_train_handle))
        else:
            _relay_pending_metrics(ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref)))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            if (x := rollout_data_next_future) is not None:
                gen_result = ray.get(x)
                if isinstance(gen_result, tuple):
                    rollout_data_curr_ref, pending = gen_result
                    _relay_pending_metrics(pending)
                else:
                    rollout_data_curr_ref = gen_result
            else:
                rollout_data_curr_ref = None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            _relay_pending_metrics(ray.get(rollout_manager.eval.remote(rollout_id)))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
