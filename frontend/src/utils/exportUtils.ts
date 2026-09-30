/**
 * exportUtils.ts
 * --------------
 * Enterprise export utility for downloading data in CSV, Excel-compatible,
 * and PDF / Printable document formats.
 */

export function exportToCSV(data: Record<string, any>[], filename: string = 'export.csv') {
  if (!data || data.length === 0) return;

  const headers = Object.keys(data[0]);
  const csvRows = [];

  // Header row
  csvRows.push(headers.join(','));

  // Data rows
  for (const row of data) {
    const values = headers.map(header => {
      const val = row[header];
      const escaped = ('' + (val ?? '')).replace(/"/g, '""');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(','));
  }

  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename.endsWith('.csv') ? filename : `${filename}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportToExcel(data: Record<string, any>[], filename: string = 'export.xls') {
  if (!data || data.length === 0) return;

  const headers = Object.keys(data[0]);
  let tsvContent = headers.join('\t') + '\n';

  for (const row of data) {
    const values = headers.map(header => row[header] ?? '');
    tsvContent += values.join('\t') + '\n';
  }

  // UTF-8 BOM for Excel compatibility
  const blob = new Blob(['\ufeff' + tsvContent], { type: 'application/vnd.ms-excel;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename.endsWith('.xls') ? filename : `${filename}.xls`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportToPDF(title: string, subtitle: string, data: Record<string, any>[]) {
  if (!data || data.length === 0) return;

  const printWindow = window.open('', '_blank');
  if (!printWindow) return;

  const headers = Object.keys(data[0]);

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <title>${title}</title>
        <style>
          body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 24px; color: #1e293b; }
          h1 { margin: 0 0 4px 0; color: #0f172a; font-size: 24px; }
          p { margin: 0 0 20px 0; color: #64748b; font-size: 13px; }
          table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
          th { background: #0f172a; color: #ffffff; text-align: left; padding: 8px 12px; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
          td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
          tr:nth-child(even) { background: #f8fafc; }
          .footer { margin-top: 30px; font-size: 10px; color: #94a3b8; text-align: right; border-top: 1px solid #e2e8f0; padding-top: 10px; }
        </style>
      </head>
      <body>
        <h1>${title}</h1>
        <p>${subtitle} • Generated on ${new Date().toLocaleString()}</p>
        <table>
          <thead>
            <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${data.map(row => `
              <tr>${headers.map(h => `<td>${row[h] ?? '-'}</td>`).join('')}</tr>
            `).join('')}
          </tbody>
        </table>
        <div class="footer">Smart Factory Predictive Maintenance Platform • Enterprise AI Report</div>
      </body>
    </html>
  `;

  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => {
    printWindow.print();
    printWindow.close();
  }, 500);
}
