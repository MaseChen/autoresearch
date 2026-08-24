import type { Meta, StoryObj } from '@storybook/react-vite'
import { Space } from 'antd'
import { StatusBadge } from './StatusBadge'

const meta = {
  title: 'Console/StatusBadge',
  component: StatusBadge,
  tags: ['autodocs'],
} satisfies Meta<typeof StatusBadge>

export default meta
type Story = StoryObj<typeof meta>

export const UnknownNoReplay: Story = { args: { value: 'UNKNOWN_GPU_OUTCOME' } }
export const Unavailable: Story = { args: { value: 'UNAVAILABLE' } }
export const StateMatrix: Story = {
  render: () => <Space wrap>{['STABLE', 'RUNNING', 'SUCCESS', 'PAUSED_OPERATOR', 'UNKNOWN_GPU_OUTCOME', 'UNAVAILABLE', 'FAILED'].map((value) => <StatusBadge key={value} value={value} />)}</Space>,
}
