import os
import csv
import json
from collections import OrderedDict


def _collect_layer_info(net):
    """Collect layer-wise information from a network.

    Returns:
        tuple: (layers, summary)
            - layers (list): List of dicts containing layer info.
            - summary (dict): Statistics including total/trainable/non-trainable params.
    """
    layers = []
    total_params = 0
    trainable_params = 0

    for name, module in net.named_modules():
        # Skip container modules to avoid duplicate parameter counting
        if list(module.children()):
            continue

        num_params = sum(p.numel() for p in module.parameters())
        total_params += num_params
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        trainable_params += trainable

        layers.append({
            'name': name,
            'type': module.__class__.__name__,
            'params': num_params,
            'trainable': trainable > 0 if num_params > 0 else None,
            'device': str(next(module.parameters()).device) if num_params > 0 else None
        })

    summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

    return layers, summary


# ==================== Built-in Exporters ====================

def export_arch_text(net, save_path, net_cls_name, device):
    """Export network architecture as plain text."""
    layers, summary = _collect_layer_info(net)

    lines = []
    lines.append('=' * 90)
    lines.append(f'Network Class: {net_cls_name}')
    lines.append(f'Device: {device}')
    lines.append('=' * 90)
    lines.append('')
    lines.append(f'{"Layer":<<45} {"Type":<<25} {"Params":>12} {"Trainable":>10}')
    lines.append('-' * 90)

    for layer in layers:
        if layer['params'] > 0:
            flag = 'True' if layer['trainable'] else 'False'
            lines.append(
                f'{layer["name"]:<45} {layer["type"]:<25} '
                f'{layer["params"]:>12,} {flag:>10}'
            )
        else:
            lines.append(
                f'{layer["name"]:<45} {layer["type"]:<25} '
                f'{"--":>12} {"--":>10}'
            )

    lines.append('-' * 90)
    lines.append(f'Total parameters:      {summary["total_params"]:>12,}')
    lines.append(f'Trainable parameters:  {summary["trainable_params"]:>12,}')
    lines.append(f'Non-trainable params:  {summary["non_trainable_params"]:>12,}')
    lines.append('=' * 90)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_arch_json(net, save_path, net_cls_name, device):
    """Export network architecture as JSON."""
    layers, summary = _collect_layer_info(net)

    data = {
        'network_class': net_cls_name,
        'device': str(device),
        'layers': layers,
        'summary': summary
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_arch_markdown(net, save_path, net_cls_name, device):
    """Export network architecture as Markdown table."""
    layers, summary = _collect_layer_info(net)

    lines = []
    lines.append(f'# Network Architecture: {net_cls_name}')
    lines.append(f'**Device:** {device}')
    lines.append('')
    lines.append('| Layer | Type | Params | Trainable |')
    lines.append('|-------|------|--------|-----------|')

    for layer in layers:
        if layer['params'] > 0:
            flag = '✓' if layer['trainable'] else '✗'
            lines.append(
                f'| `{layer["name"]}` | {layer["type"]} | '
                f'{layer["params"]:,} | {flag} |'
            )
        else:
            lines.append(
                f'| `{layer["name"]}` | {layer["type"]} | -- | -- |'
            )

    lines.append('')
    lines.append('## Summary')
    lines.append(f'- **Total parameters:** {summary["total_params"]:,}')
    lines.append(f'- **Trainable parameters:** {summary["trainable_params"]:,}')
    lines.append(f'- **Non-trainable parameters:** {summary["non_trainable_params"]:,}')

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_arch_csv(net, save_path, net_cls_name, device):
    """Export network architecture as CSV."""
    layers, summary = _collect_layer_info(net)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Network Class', net_cls_name])
        writer.writerow(['Device', str(device)])
        writer.writerow([])
        writer.writerow(['Layer', 'Type', 'Params', 'Trainable'])

        for layer in layers:
            if layer['params'] > 0:
                flag = 'True' if layer['trainable'] else 'False'
                writer.writerow([
                    layer['name'], layer['type'],
                    layer['params'], flag
                ])
            else:
                writer.writerow([layer['name'], layer['type'], '--', '--'])

        writer.writerow([])
        writer.writerow(['Summary', '', '', ''])
        writer.writerow(['Total', summary['total_params'], '', ''])
        writer.writerow(['Trainable', summary['trainable_params'], '', ''])
        writer.writerow(['Non-trainable', summary['non_trainable_params'], '', ''])


# ==================== Registry ====================

ARCH_EXPORTERS = {
    'txt': export_arch_text,
    'json': export_arch_json,
    'md': export_arch_markdown,
    'csv': export_arch_csv,
}


def get_exporter(fmt):
    """Retrieve an exporter by format name.

    Args:
        fmt (str): Format identifier (e.g., 'txt', 'json', 'md', 'csv').

    Returns:
        callable: The exporter function.

    Raises:
        ValueError: If the format is not registered.
    """
    if fmt not in ARCH_EXPORTERS:
        raise ValueError(
            f"Unknown format: {fmt}. "
            f"Supported: {list(ARCH_EXPORTERS.keys())}"
        )
    return ARCH_EXPORTERS[fmt]


def register_exporter(fmt, fn):
    """Register a custom exporter.

    Args:
        fmt (str): Format identifier.
        fn (callable): Exporter function with signature
            (net, save_path, net_cls_name, device).
    """
    ARCH_EXPORTERS[fmt] = fn