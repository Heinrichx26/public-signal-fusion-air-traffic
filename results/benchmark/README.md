# Airport-hour fusion benchmark

This benchmark defines reusable airport-hour tasks for multi-source information fusion in air traffic disruptions. It uses flight outcomes, surface weather, schedule-derived demand, ATCSCC advisory signals, and realized outcomes.

Core tasks:
- residual state detection;
- long-delay prediction;
- cancellation prediction;
- post-advisory persistence detection;
- event-level advisory validation.

The benchmark tables define fields, task targets, split rules, and baseline scores. Plotting and reported tables read these result tables directly.
